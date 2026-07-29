# Bean Disease Classification ML Pipeline

## Project Description
This project implements an end-to-end machine learning pipeline for classifying bean leaf diseases from images. The system includes:

- Image-based data preprocessing
- Deep learning model training and evaluation
- Prediction API for single-image inference
- Bulk upload and retraining workflow
- Streamlit dashboard for prediction, visualization, and retraining controls
- Docker-based deployment setup
- Load testing with Locust

The model classifies bean leaf images into three classes:
- Angular Leaf Spot
- Bean Rust
- Healthy

## Project Structure
```text
bean-disease-ml-pipeline/
├── app.py
├── docker-compose.yml
├── Dockerfile
├── locustfile.py
├── requirements.txt
├── api/
│   └── main.py
├── dashboard/
├── data/
│   └── train/
├── models/
│   └── bean_model.h5
├── notebook/
│   └── Bean_Disease_Classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   ├── retrain.py
│   ├── train.py
│   └── visualization.py
└── uploads/
```

## Features
- Upload a single leaf image and get a predicted disease class
- View dataset visualizations and feature interpretations
- Upload a ZIP file containing new images for retraining
- Trigger model retraining from the dashboard
- Run the API and dashboard locally or with Docker
- Simulate traffic using Locust for performance testing

## Demo Video
Add your YouTube video link here:

- Demo Video: https://your-demo-video-link

## Deployment URL
If deployed to a cloud platform, add the live URL here:

Backend API (Render):
[https://bean-disease1.onrender.com/](https://bean-disease1.onrender.com)

Swagger Documentation:
[https://bean-disease1.onrender.com/docs](https://bean-disease-streamlit-y9mi.onrender.com)

Frontend (Streamlit):
[https://YOUR-STREAMLIT-URL.onrender.com/](https://beandisease-ui.onrender.com)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ulrichr7/bean_disease.git
cd bean-disease
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Run the Dashboard
```bash
streamlit run app.py
```

### 6. Run with Docker
```bash
docker-compose up --build
```

The API will be available at:
- http://localhost:8000

The Streamlit dashboard will be available at:
- http://localhost:8501

## Notebook
The notebook for model training and evaluation is located at:
- [notebook/Bean_Disease_Classification.ipynb](notebook/Bean_Disease_Classification.ipynb)

It includes:
- Data preprocessing steps
- Model training
- Model evaluation
- Prediction and testing functions

## Model File
The trained model is stored at:
- [models/bean_model.h5](models/bean_model.h5)

## Retraining Workflow
To retrain the model:
1. Upload a ZIP archive containing new images through the dashboard.
2. The uploaded files are saved into the training data folders.
3. Click the retraining button in the dashboard.
4. The API retrains the existing model using the new data.

## Load Testing
To run flood/request simulation using Locust:
```bash
locust -f locustfile.py --host http://localhost:8001
```

Then open the Locust UI in your browser and start a test.

## Sample Load Test Results:

10 users
Average Response Time: 83 ms

50 users
Average Response Time: 112 ms

100 users
Average Response Time: 171 ms

Failure Rate: 0%

## Notes
- The project uses TensorFlow/Keras for image classification.
- The API uses FastAPI for prediction and retraining endpoints.
- The dashboard uses Streamlit for user interaction.
- Docker is used for easy deployment and scaling.

## Monitoring / Metrics
The API exposes a Prometheus-compatible metrics endpoint at `/metrics` (port 8000/8001 depending on your run). To collect metrics:

- Install and configure Prometheus to scrape `http://<host>:8000/metrics`.
- Optionally add Grafana to visualize `bean_api_requests_total` and `bean_api_request_latency_seconds`.

Example Prometheus scrape config snippet:
```
scrape_configs:
	- job_name: 'bean_api'
		static_configs:
			- targets: ['localhost:8000']
```

### Scaling locally with Docker Compose
You can run multiple replicas of the API to simulate scaling (Docker Engine and Compose v2 required):

```powershell
# build once
docker-compose up --build -d

# scale service to 3 replicas
docker compose up --scale bean-api=3 -d
```

When running multiple replicas, put a load-balancer or use Docker's internal routing to distribute traffic across containers. For simple local tests, Locust will send requests to the compose-exposed port and Docker's routing will round-robin.

### Run Prometheus + Grafana locally (monitoring stack)
There's a small compose file to launch Prometheus and Grafana for local scraping and visualization.

```powershell
docker compose -f docker-compose.monitor.yml up -d
```

- Prometheus UI: http://localhost:9090
- Grafana UI: http://localhost:3000 (default credentials: admin/admin)

The provided Grafana provisioning auto-adds the Prometheus datasource.

## Headless Load Testing
Run Locust headless to produce reproducible CSV outputs. Use the included PowerShell script:

```powershell
.\run_locust.ps1 -users 200 -spawnRate 20 -runTime "2m" -host "http://localhost:8001"
```

This will produce CSV files named like `locust-results-<timestamp>_stats.csv` which you can inspect for average/median response times and failures.

## Model Evaluation

The CNN model was evaluated using the test dataset.

Evaluation Metrics:

- Accuracy: 94.37%
- Precision: 94.33%
- Recall: 94.37%
- F1-Score: 94.34%

The notebook includes the confusion matrix, classification report, and training history.

## Technologies Used

- Python 3.11
- TensorFlow / Keras
- FastAPI
- Streamlit
- Docker
- Render
- Prometheus
- Grafana
- Locust
- OpenCV
- NumPy
- Pandas


## Notes and recommendations
- Currently uploads are saved to disk under `data/train/<label>/`. If you require saving uploads to a database for auditability, add a simple record-keeping DB (SQLite/Postgres) and update `/upload-bulk` to write metadata to the DB.
- Retraining overwrites `models/bean_model.h5`. Consider versioning models with timestamps or hashes (e.g., `bean_model_20260728.h5`) for rollback.

## License
This project is for educational purposes.
