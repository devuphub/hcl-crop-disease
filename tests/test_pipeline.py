"""
Comprehensive Test Suite for Crop Disease Detection System
Tests CNN Architecture, Predictor Pipeline, Grad-CAM, PDF Generator, and REST API.
"""

import os
import sys
import unittest
from io import BytesIO
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import CropDiseaseCNN, get_model
from models.gradcam import GradCAM
from models.predictor import CropDiseasePredictor
from src.dataset_builder import _create_base_leaf, _apply_disease_symptoms
from src.report_generator import generate_pdf_report
from fastapi.testclient import TestClient
from api.main import app


class TestCropDiseasePipeline(unittest.TestCase):

    def test_01_cnn_model_forward(self):
        """Test CNN model layer instantiation and forward pass tensor shapes"""
        model = CropDiseaseCNN(num_classes=18)
        model.eval()
        dummy_tensor = torch.randn(2, 3, 224, 224)
        output = model(dummy_tensor)
        self.assertEqual(output.shape, (2, 18), "Output tensor shape should be (2, 18)")

    def test_02_mobilenet_forward(self):
        """Test MobileNetV2 architecture forward pass"""
        model = get_model("mobilenet", num_classes=18, pretrained=False)
        model.eval()
        dummy_tensor = torch.randn(1, 3, 224, 224)
        output = model(dummy_tensor)
        self.assertEqual(output.shape, (1, 18))

    def test_03_leaf_synthesis(self):
        """Test synthetic leaf rendering pipeline"""
        base_leaf = _create_base_leaf("Tomato", 224, 224)
        self.assertEqual(base_leaf.size, (224, 224))
        diseased = _apply_disease_symptoms(base_leaf, "Tomato___Early_blight")
        self.assertEqual(diseased.size, (224, 224))

    def test_04_predictor_inference(self):
        """Test end-to-end predictor pipeline on PIL image"""
        test_img = Image.new("RGB", (224, 224), color=(30, 150, 40))
        predictor = CropDiseasePredictor()
        result = predictor.predict(test_img, top_k=3, generate_heatmap=True)

        self.assertIn("primary_prediction", result)
        self.assertIn("top_predictions", result)
        self.assertIn("severity", result)
        self.assertIn("details", result)
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertGreaterEqual(result["primary_prediction"]["confidence"], 0)

    def test_05_gradcam_overlay(self):
        """Test Explainable AI Grad-CAM generation"""
        model = CropDiseaseCNN(num_classes=18)
        grad_cam = GradCAM(model)
        dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        heatmap = grad_cam.generate_cam(dummy_tensor, target_class=0)
        
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(heatmap.max(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)
        
        test_img = Image.new("RGB", (224, 224), color=(50, 180, 50))
        overlay = grad_cam.overlay_heatmap(test_img, heatmap)
        self.assertEqual(overlay.size, (224, 224))
        grad_cam.remove_hooks()

    def test_06_pdf_report_generation(self):
        """Test PDF advisory report generator"""
        sample_diagnosis = {
            "primary_prediction": {
                "crop": "Tomato",
                "disease": "Early Blight",
                "confidence": 94.8,
                "is_healthy": False
            },
            "severity": "Moderate to High",
            "details": {
                "pathogen": "Alternaria solani",
                "cause": "Warm temperatures (24-29°C) and alternating wet and dry weather.",
                "symptoms": "Brown to black lesions with concentric target rings.",
                "organic_treatment": ["Spray Bacillus subtilis."],
                "chemical_treatment": ["Mancozeb 75 WP."],
                "prevention": ["Crop rotation with non-solanaceous crops."]
            }
        }
        pdf_bytes = generate_pdf_report(sample_diagnosis)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000, "PDF buffer must contain content")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Must be a valid PDF document header")

    def test_07_fastapi_endpoints(self):
        """Test REST API /health, /classes, and /predict endpoints"""
        client = TestClient(app)
        
        # Health check
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        # Classes
        res_classes = client.get("/classes")
        self.assertEqual(res_classes.status_code, 200)
        self.assertEqual(res_classes.json()["total_classes"], 18)

        # Predict
        buf = BytesIO()
        test_img = Image.new("RGB", (224, 224), color=(40, 160, 40))
        test_img.save(buf, format="JPEG")
        buf.seek(0)

        res_predict = client.post(
            "/predict",
            files={"file": ("test_leaf.jpg", buf, "image/jpeg")},
            data={"top_k": 3}
        )
        self.assertEqual(res_predict.status_code, 200)
        data = res_predict.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("primary_prediction", data)


if __name__ == "__main__":
    unittest.main()
