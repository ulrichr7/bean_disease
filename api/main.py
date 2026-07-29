import os
import io
import zipfile
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, JSONResponse
import uuid
import time
from datetime import datetime
from . import db

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TENSORFLOW_IMPORT_ERROR = exc
else:
    TENSORFLOW_IMPORT_ERROR = None

app = FastAPI(title="Production Bean ML Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mandatory Directory Layout Resolvers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/bean_model.h5")
UPLOAD_DIR = os.path.join(BASE_DIR, "../data/train")

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure fallback label folders exist in the upload tree
for label in CLASS_NAMES:
    os.makedirs(os.path.join(UPLOAD_DIR, label), exist_ok=True)

# Load global inference engine model
if tf is None:
    model = None
    print(f"⚠️ TensorFlow unavailable at startup: {TENSORFLOW_IMPORT_ERROR}")
elif os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("🚀 Base model loaded successfully.")
else:
    model = None
    print("⚠️ Base model file missing at script initialization.")

# Initialize DB for uploads / model versions
try:
    db.init_db()
except Exception:
    print("Failed initializing metadata DB")

# Prometheus metrics
REQUEST_COUNT = Counter(
    'bean_api_requests_total', 'Total number of requests to the Bean API', ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'bean_api_request_latency_seconds', 'Latency of requests to the Bean API', ['endpoint']
)

# In-memory task tracker for long-running background jobs
TASKS = {}


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise
    finally:
        resp_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(resp_time)
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=str(status_code)).inc()
    return response


@app.get('/metrics')
def metrics():
    """Prometheus scrape endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """Preprocesses a single image byte stream to a normalized NumPy array."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    return np.array(image) / 255.0

@app.get("/")
def read_root():
    return {"status": "healthy", "pipeline": "Active"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        if tf is None:
            raise HTTPException(status_code=500, detail=f"Inference engine unavailable: {TENSORFLOW_IMPORT_ERROR}")
        raise HTTPException(status_code=500, detail="Inference engine uninitialized.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    try:
        contents = await file.read()
        input_tensor = np.expand_dims(preprocess_image(contents), axis=0)
        predictions = model.predict(input_tensor)
        flat_predictions = np.squeeze(predictions)

        if flat_predictions.ndim == 0:  # Binary fallback
            raw_score = float(flat_predictions)
            predicted_class_idx = 1 if raw_score > 0.5 else 0
            confidence_score = raw_score if predicted_class_idx == 1 else (1.0 - raw_score)
        else:  # Multi-class
            predicted_class_idx = int(np.argmax(flat_predictions))
            confidence_score = float(flat_predictions[predicted_class_idx])

        return {"class": CLASS_NAMES[predicted_class_idx], "confidence": confidence_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-bulk")
async def upload_bulk(file: UploadFile = File(...)):
    """Rubric Req: Bulk Data Uploading + Saving to Database/Disk."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Please upload a .zip archive of dataset image files.")
    try:
        contents = await file.read()
        zip_buffer = io.BytesIO(contents)
        saved_count = 0
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue  # skip directories

                # Automatically route images into label directory structures based on filename keywords
                assigned_label = "healthy"
                if "rust" in filename.lower():
                    assigned_label = "bean_rust"
                elif "angular" in filename.lower() or "spot" in filename.lower():
                    assigned_label = "angular_leaf_spot"

                save_path = os.path.join(UPLOAD_DIR, assigned_label, filename)
                with open(save_path, "wb") as f:
                    f.write(zip_ref.read(member))
                saved_count += 1
                # Record in metadata DB
                try:
                    db.record_upload(filename=filename, label=assigned_label, saved_path=save_path)
                except Exception:
                    # non-fatal: continue
                    pass
                    
        return {"status": "success", "message": f"Successfully unzipped and saved {saved_count} batch artifacts to {UPLOAD_DIR}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk filesystem write failed: {str(e)}")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Schedule retraining as a background job to avoid blocking the API."""
    global model, MODEL_PATH
    if tf is None:
        raise HTTPException(status_code=500, detail=f"Retraining unavailable: {TENSORFLOW_IMPORT_ERROR}")
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Base model weights file not found to use as pre-trained baseline.")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "scheduled", "created_at": datetime.utcnow().isoformat(), "message": None}
    background_tasks.add_task(_retrain_background, task_id)
    return JSONResponse(status_code=202, content={"status": "scheduled", "task_id": task_id, "message": "Retraining started in background."})


def _retrain_background(task_id: str):
    """Background worker that performs the retraining and updates TASKS."""
    global model, MODEL_PATH
    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["started_at"] = datetime.utcnow().isoformat()
    try:
        # Load pre-trained baseline
        local_model = tf.keras.models.load_model(MODEL_PATH)

        # Preprocess uploaded images
        images, labels = [], []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_folder = os.path.join(UPLOAD_DIR, class_name)
            if not os.path.isdir(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                try:
                    with open(img_path, "rb") as f:
                        img_array = preprocess_image(f.read())
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception:
                    continue

        if len(images) < 2:
            TASKS[task_id].update({"status": "error", "message": "Insufficient training samples."})
            return

        X_train = np.array(images)
        y_train = np.array(labels)

        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        local_model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)

        models_dir = os.path.join(BASE_DIR, "../models")
        os.makedirs(models_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        new_model_filename = f"bean_model_{ts}.h5"
        new_model_path = os.path.join(models_dir, new_model_filename)
        local_model.save(new_model_path)

        model = local_model
        MODEL_PATH = new_model_path

        try:
            db.record_model_version(model_path=new_model_path, samples=len(images))
        except Exception:
            pass

        TASKS[task_id].update({"status": "completed", "finished_at": datetime.utcnow().isoformat(), "model": new_model_filename, "samples": len(images)})
    except Exception as e:
        TASKS[task_id].update({"status": "error", "message": str(e)})


@app.get("/models")
def list_model_versions():
    try:
        rows = db.list_models()
        models = [ {"id": r[0], "model_path": r[1], "created_at": r[2], "samples": r[3]} for r in rows ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrain/status/{task_id}")
def retrain_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
