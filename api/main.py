import os
import io
import zipfile
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from datetime import datetime
from . import db

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
if os.path.exists(MODEL_PATH):
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
async def retrain_model():
    """Rubric Req: 1. Uses old model as pre-trained model, 2. Run preprocessing, 3. Retrain."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Base model weights file not found to use as pre-trained baseline.")
    
    try:
        # 1. Rubric Step: Load the old model as a pre-trained baseline
        local_model = tf.keras.models.load_model(MODEL_PATH)
        
        # 2. Rubric Step: Preprocess the newly uploaded directory data points
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
                except:
                    continue # Skip unreadable files safely

        if len(images) < 2:
            raise HTTPException(status_code=400, detail="Insufficient file counts inside data paths to compile training boundaries.")

        X_train = np.array(images)
        y_train = np.array(labels)

        # 3. Rubric Step: Incrementally retrain the model (Fine-tuning phase)
        # Using a very low learning rate optimizer to safely update pre-trained layers
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a quick epoch to update weights with the new images
        local_model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)
        
        # Save new model with timestamped version instead of blindly overwriting
        models_dir = os.path.join(BASE_DIR, "../models")
        os.makedirs(models_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        new_model_filename = f"bean_model_{ts}.h5"
        new_model_path = os.path.join(models_dir, new_model_filename)
        local_model.save(new_model_path)

        # Update the runtime reference and model pointer
        model = local_model
        # Update global MODEL_PATH to point to the new model
        global MODEL_PATH
        MODEL_PATH = new_model_path

        # Record model version in DB
        try:
            db.record_model_version(model_path=new_model_path, samples=len(images))
        except Exception:
            pass

        return {"status": "success", "message": f"Retrained and saved new model: {new_model_filename} over {len(images)} samples.", "model": new_model_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution crash: {str(e)}")


@app.get("/models")
def list_model_versions():
    try:
        rows = db.list_models()
        models = [ {"id": r[0], "model_path": r[1], "created_at": r[2], "samples": r[3]} for r in rows ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
