import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Import the exact function from your model.py
from model import get_model 
from inference import get_classifier

# Standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_images(img_array, sigmaX=10):
    image = img_array.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = image[y:y+h, x:x+w]
        
    clean_display_img = cv2.resize(image, (224, 224))
    rgb_image = cv2.cvtColor(clean_display_img, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(rgb_image, (0, 0), sigmaX)
    model_input_img = cv2.addWeighted(rgb_image, 4, blurred, -4, 128)
    
    return model_input_img

def main():
    img_array = np.zeros((300, 300, 3), dtype=np.uint8) + 128 # gray image
    
    # Simulate FastAPI flow
    model_input_array = preprocess_images(img_array)
    processed_pil = Image.fromarray(model_input_array)
    input_tensor = transform(processed_pil).unsqueeze(0).to(device)

    # 1. Test model.py approach (similar to main.py)
    print("--- Testing model.py (main.py approach) ---")
    model_main = get_model() 
    try:
        state_dict = torch.load('Resnet-50/resnet50_best_model.pth', map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model_main.load_state_dict(new_state_dict, strict=False)
        model_main.to(device)
        model_main.eval()
        print("Model loaded.")
        with torch.no_grad():
            output_main = model_main(input_tensor)
            probs_main = torch.nn.functional.softmax(output_main[0], dim=0)
            print(f"Output probs: {probs_main}")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        
    # 2. Test inference.py approach
    print("\n--- Testing inference.py approach ---")
    classifier = get_classifier()
    print("Model initialized.")
    # In `inference.py`, it loads pre-trained ImageNet weights (unless model_weights_path is passed)
    # the classifier instance used in main application currently does NOT pass model_weights_path!
    try:
        with torch.no_grad():
            output_inf = classifier.model(input_tensor)
            probs_inf = torch.nn.functional.softmax(output_inf[0], dim=0)
            print(f"Output probs: {probs_inf}")
    except Exception as e:
        print(f"Inference error: {e}")
        
if __name__ == "__main__":
    main()
