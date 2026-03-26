import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNet50Inference:
    """
    A production-ready PyTorch inference pipeline using ResNet50 for 5-class image classification.
    Structured to be easily integrated into FastAPI or other web frameworks.
    """
    def __init__(self, num_classes: int = 5, model_weights_path: str = None):
        """
        Initialize the model, device, and preprocessing pipeline.
        
        Args:
            num_classes (int): Number of output classes (5).
            model_weights_path (str, optional): Path to custom fine-tuned weights (.pth).
                                                If None, uses ImageNet weights.
        """
        # Determine the available device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ResNet50 with pretrained ImageNet weights
        try:
            # PyTorch 1.13+ recommended approach
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.model = models.resnet50(pretrained=True)
            
        # Replace the final fully connected layer for 5-class classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load custom fine-tuned weights if provided
        if model_weights_path and os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path, map_location=self.device)
            # Remove 'module.' prefix if the model was trained with DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
            
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define standard ImageNet preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single image and return predicted class and confidence.
        
        Args:
            image_path (str): Local path to the image file.
            
        Returns:
            dict: Contains 'predicted_class' (0-4), 'confidence' (float), and 'status'.
        """
        # Clean error handling: Check if file exists
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found at path: {image_path}",
                "status": "failed"
            }
            
        # Clean error handling: Catch bad image files
        try:
            # Ensure image is RGB (handles PNGs with alpha channels, grayscale, etc.)
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {
                "error": f"Failed to open or process image: {str(e)}",
                "status": "failed"
            }
            
        try:
            import cv2
            import numpy as np
            
            # Convert PIL image to BGR openCV format
            raw_img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # --- BEGIN PREPROCESSING FROM MAIN.PY ---
            sigmaX = 10
            img_copy = raw_img_array.copy()
            
            # Crop the black borders to isolate the circular retina
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                img_copy = img_copy[y:y+h, x:x+w]
                
            clean_display_img = cv2.resize(img_copy, (224, 224))
            rgb_image = cv2.cvtColor(clean_display_img, cv2.COLOR_BGR2RGB)
            blurred = cv2.GaussianBlur(rgb_image, (0, 0), sigmaX)
            model_input_img = cv2.addWeighted(rgb_image, 4, blurred, -4, 128)
            # --- END PREPROCESSING ---
            
            processed_pil = Image.fromarray(model_input_img)
            
            # Apply preprocessing and add batch dimension -> [1, 3, 224, 224]
            input_tensor = self.transform(processed_pil).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Disable gradient calculation for faster memory-efficient inference
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # --- BOOST DR SENSITIVITY ---
                # The dataset was highly imbalanced, causing mostly "No DR" predictions.
                # We penalize "No DR" (Class 0) by -2.0 and slightly boost the DR classes to force sensitivity.
                # You can tweak these values to make the model more or less sensitive.
                sensitivity = torch.tensor([-2.0, 0.5, 0.5, 0.5, 0.5]).to(self.device)
                output = output + sensitivity
                
                # Apply softmax to calculate probabilities
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # Retrieve the class with the highest probability
                confidence, predicted_class = torch.max(probabilities, dim=0)
                
                return {
                    "predicted_class": int(predicted_class.item()),
                    "confidence": float(confidence.item()),
                    "status": "success"
                }
                
        except Exception as e:
            return {
                "error": f"Inference execution failed: {str(e)}",
                "status": "failed"
            }

# ------------------------------------------------------------------------------
# Global Singletons & Wrapper Functions (Ideal for FastAPI Integration)
# ------------------------------------------------------------------------------

# Singleton instance to avoid reloading the model into memory per request
_CLASSIFIER_INSTANCE = None

def get_classifier() -> ResNet50Inference:
    """
    Returns a global instance of the ResNet50Inference class.
    Perfect for FastAPI dependency injection: `Depends(get_classifier)`
    """
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = ResNet50Inference(
            num_classes=5, 
            model_weights_path='Resnet-50/resnet50_best_model.pth'
        )
    return _CLASSIFIER_INSTANCE

def predict_image(image_path: str) -> dict:
    """
    Wrapper function to directly predict an image class from a local path.
    """
    classifier = get_classifier()
    return classifier.predict(image_path)


if __name__ == "__main__":
    # Example command-line usage:
    # python inference.py path/to/sample_image.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Running inference on: {img_path}")
        result = predict_image(img_path)
        print("Result:", result)
    else:
        print("Usage: python inference.py <path_to_image>")
