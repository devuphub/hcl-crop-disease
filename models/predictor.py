"""
Crop Disease Predictor Engine
Performs model inference, top-k class prediction, agronomic diagnosis lookup,
and Grad-CAM heatmap generation.
"""

import os
import json
from typing import Dict, Any, List, Tuple
from PIL import Image
import torch
from torchvision import transforms

from models.cnn_model import get_model
from models.gradcam import GradCAM


class CropDiseasePredictor:
    """
    Inference and diagnosis pipeline for Crop Disease Detection.
    """
    def __init__(self,
                 model_path: str = None,
                 class_names_path: str = None,
                 disease_info_path: str = None,
                 device: str = None):
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.model_path = model_path or os.path.join(base_dir, "models", "saved_models", "crop_disease_model.pth")
        self.class_names_path = class_names_path or os.path.join(base_dir, "models", "saved_models", "class_names.json")
        self.disease_info_path = disease_info_path or os.path.join(base_dir, "data", "disease_info.json")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load class mappings
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
                "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                "Rice___Brown_Spot", "Rice___Leaf_Blast", "Rice___healthy",
                "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
                "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___healthy"
            ]

        # Load disease info knowledge base
        if os.path.exists(self.disease_info_path):
            with open(self.disease_info_path, "r", encoding="utf-8") as f:
                self.disease_info = json.load(f)
        else:
            self.disease_info = {}

        # Standard preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        num_classes = len(self.class_names)
        
        # Check if saved model specifies architecture
        model_type = "cnn"
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_type" in checkpoint:
                model_type = checkpoint["model_type"]
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            model = get_model(model_type, num_classes=num_classes, pretrained=False)
            model.load_state_dict(state_dict)
        else:
            # Fallback initialized model if checkpoint not yet generated
            model = get_model(model_type, num_classes=num_classes, pretrained=False)

        return model

    def predict(self, image: Image.Image, top_k: int = 3, generate_heatmap: bool = True) -> Dict[str, Any]:
        """
        Runs prediction on a PIL Image and returns structured diagnostic information.
        """
        # Ensure RGB
        img_rgb = image.convert("RGB")
        tensor_img = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        
        top_predictions: List[Dict[str, Any]] = []
        for prob, idx in zip(top_probs, top_indices):
            c_name = self.class_names[idx.item()]
            info = self.disease_info.get(c_name, {})
            crop_name = info.get("crop", c_name.split("___")[0])
            disease_name = info.get("disease", c_name.split("___")[1].replace("_", " "))
            
            top_predictions.append({
                "class_id": idx.item(),
                "class_name": c_name,
                "crop": crop_name,
                "disease": disease_name,
                "confidence": round(prob.item() * 100, 2),
                "is_healthy": info.get("is_healthy", "healthy" in c_name.lower())
            })

        best = top_predictions[0]
        disease_metadata = self.disease_info.get(best["class_name"], {
            "crop": best["crop"],
            "disease": best["disease"],
            "is_healthy": best["is_healthy"],
            "pathogen": "Unknown",
            "severity": "Unknown",
            "symptoms": "No specific record found.",
            "cause": "Unknown",
            "organic_treatment": ["Isolate affected leaves and monitor."],
            "chemical_treatment": ["Consult local agricultural extension service."],
            "prevention": ["Maintain clean cultivation practices."]
        })

        # Calculate severity rating
        confidence_val = best["confidence"]
        if best["is_healthy"]:
            status_badge = "Healthy"
            severity_level = "None"
        elif confidence_val > 80:
            severity_level = disease_metadata.get("severity", "High")
            status_badge = "High Alert"
        elif confidence_val > 50:
            severity_level = "Moderate"
            status_badge = "Moderate Alert"
        else:
            severity_level = "Low / Uncertain"
            status_badge = "Low Confidence"

        result = {
            "primary_prediction": best,
            "top_predictions": top_predictions,
            "status": status_badge,
            "severity": severity_level,
            "details": disease_metadata
        }

        # Generate Grad-CAM if requested
        if generate_heatmap:
            try:
                grad_cam = GradCAM(self.model)
                tensor_input = self.transform(img_rgb).unsqueeze(0).to(self.device)
                tensor_input.requires_grad = True
                heatmap = grad_cam.generate_cam(tensor_input, target_class=best["class_id"])
                overlay_img = grad_cam.overlay_heatmap(img_rgb, heatmap, alpha=0.45)
                grad_cam.remove_hooks()
                result["cam_overlay"] = overlay_img
            except Exception as e:
                result["cam_error"] = str(e)

        return result
