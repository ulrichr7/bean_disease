from locust import HttpUser, task, between
import os
import io
from PIL import Image

class MachineLearningAPIUser(HttpUser):
    # Simulates users waiting between 1 and 3 seconds between requests
    wait_time = between(1, 3)

    @task(1)
    def test_health_check_endpoint(self):
        """Pings the root route to evaluate infrastructure baseline delivery latency."""
        self.client.get("/")

    @task(2)
    def test_prediction_payload_flood(self):
        """Simulates a heavy multi-user upload flood hitting the prediction route."""
        # 1. Create a valid RGB image in memory matching standard ML model expectations
        img = Image.new('RGB', (256, 256), color='green')
        
        # 2. Convert the image data into a stream of raw bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        payload_bytes = img_byte_arr.getvalue()

        files = {
            "file": ("simulation_leaf.png", payload_bytes, "image/png")
        }

        # 3. POST the functional payload directly to your route endpoint
        self.client.post("/predict", files=files)
