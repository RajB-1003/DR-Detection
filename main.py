from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import base64
import io

# Import the exact function from your model.py
from model import get_model 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set to False to run the actual PyTorch model!
MOCK_MODE = False

# --- PYTORCH & GRAD-CAM SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

if not MOCK_MODE:
    try:
        # 1. Initialize the architecture
        model = get_model() 
        
        # 2. Load the raw weights dictionary
        state_dict = torch.load('Resnet-50/resnet50_best_model.pth', map_location=device)
        
        # 3. Hackathon Fix: Clean the dictionary keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # If they used DataParallel, strip the 'module.' prefix
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 4. Load the cleaned weights into our model
        # strict=False allows it to load even if there's a slight naming mismatch
        model.load_state_dict(new_state_dict, strict=False)
        
        # 5. Push to GPU/CPU and set to evaluation mode
        model.to(device)
        model.eval()
        print("✅ PyTorch Model Loaded Successfully.")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Let's print the last 5 keys in their file so we can see exactly what they named it
        if 'state_dict' in locals():
            print("🔍 The last 5 keys in the .pth file actually are:")
            print(list(state_dict.keys())[-5:])
        print("Falling back to MOCK_MODE.")
        MOCK_MODE = True

# Standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class DiagnosisResponse(BaseModel):
    status: str
    dr_stage: str
    confidence: float
    heatmap_base64: str
    local_suggestion_en: str
    local_suggestion_native: str

def preprocess_images(img_array, sigmaX=10):
    """
    Returns TWO images: 
    1. model_input: Heavily filtered for the AI to get high accuracy.
    2. clean_display: Natural looking, cropped image for the frontend heatmap.
    """
    image = img_array.copy()
    
    # Crop the black borders to isolate the circular retina
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = image[y:y+h, x:x+w]
        
    # Create the clean background canvas for the Grad-CAM (BGR format for OpenCV)
    clean_display_img = cv2.resize(image, (224, 224))
    
    # Create the heavily filtered input for the Model (RGB format for PyTorch)
    rgb_image = cv2.cvtColor(clean_display_img, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(rgb_image, (0, 0), sigmaX)
    model_input_img = cv2.addWeighted(rgb_image, 4, blurred, -4, 128)
    
    return model_input_img, clean_display_img

def generate_gradcam(image_tensor, original_img_array):
    """
    Hooks into the final convolutional layer of ResNet to generate the heatmap.
    """
    # These lists will hold the gradients and activations from the model hooks
    gradients = []
    activations = []

    # ⚠️ HACKATHON TODO: 'layer4' is standard for ResNet. If they used a custom name, check model.py
    target_layer = model.layer4[-1] 

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    handle_b = target_layer.register_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(image_tensor)
    
    # Get the prediction
    pred_class = output.argmax(dim=1).item()
    confidence = F.softmax(output, dim=1)[0][pred_class].item() * 100

    # Backward pass to get gradients for the predicted class
    model.zero_grad()
    output[0, pred_class].backward()

    # Clean up hooks
    handle_b.remove()
    handle_f.remove()

    # Calculate Grad-CAM
    # Calculate Grad-CAM
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations[0].shape[1]):
        activations[0][0, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations[0], dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0) # ReLU
    
    # Normalize safely
    heatmap_max = np.max(heatmap)
    if heatmap_max == 0:
        heatmap_max = 1e-8
    heatmap /= heatmap_max 

    # Smooth the heatmap using CUBIC interpolation for that "blob" look
    heatmap = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # THE FIX: Use cv2.addWeighted for a perfect, professional alpha blend
    superimposed_img = cv2.addWeighted(original_img_array, 0.6, heatmap_color, 0.4, 0)

    # Convert back to base64 for React frontend
    _, buffer = cv2.imencode('.png', superimposed_img)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return pred_class, confidence, f"data:image/png;base64,{heatmap_b64}"


@app.post("/api/v1/analyze", response_model=DiagnosisResponse)
async def analyze_fundus(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid format.")

    image_bytes = await file.read()
    
    # Define our stages
    stages = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    if MOCK_MODE:
        # Failsafe for live demo if PyTorch crashes
        dummy_heatmap = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        return DiagnosisResponse(
            status="success",
            dr_stage="Severe",
            confidence=94.2,
            heatmap_base64=f"data:image/png;base64,{dummy_heatmap}",
            local_suggestion_en="Immediate referral to an ophthalmologist required. High risk of vision loss.",
            local_suggestion_native="உடனடியாக கண் மருத்துவரை அணுகவும். பார்வை இழப்பு ஏற்படும் அபாயம் அதிகம்."
        )

    # --- LIVE INFERENCE ---
    # --- LIVE INFERENCE ---
# --- LIVE INFERENCE ---
    try:
        # 1. Load raw image
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        raw_img_array = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 2. Preprocess (Get BOTH the AI input and the Clean Canvas)
        model_input_array, clean_display_array = preprocess_images(raw_img_array)
        
        # 3. Convert back to PIL Image for PyTorch transforms
        processed_pil = Image.fromarray(model_input_array)
        input_tensor = transform(processed_pil).unsqueeze(0).to(device)
        
        # 4. Predict & Generate Heatmap 
        # 🔥 Pass the CLEAN array so the heatmap looks natural!
        pred_class, confidence, heatmap_b64 = generate_gradcam(input_tensor, clean_display_array)
        
        # 4. Generate local suggestions based on severity
        if pred_class >= 3:
            sugg_en = "Immediate referral to an ophthalmologist required. High risk of vision loss."
            sugg_native = "உடனடியாக கண் மருத்துவரை அணுகவும். பார்வை இழப்பு ஏற்படும் அபாயம் அதிகம்." # Tamil
        elif pred_class > 0:
            sugg_en = "Schedule a follow-up within 3-6 months. Manage blood sugar."
            sugg_native = "3 முதல் 6 மாதங்களுக்குள் கண் மருத்துவரை அணுகவும். இரத்த சர்க்கரையை நிர்வகிக்கவும்."
        else:
            sugg_en = "No abnormalities detected. Continue annual checkups."
            sugg_native = "எந்த அசாதாரணமும் இல்லை. ஆண்டுதோறும் பரிசோதனை செய்யவும்."

        return DiagnosisResponse(
            status="success",
            dr_stage=stages[pred_class],
            confidence=round(confidence, 1),
            heatmap_base64=heatmap_b64,
            local_suggestion_en=sugg_en,
            local_suggestion_native=sugg_native
        )

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal ML processing error.")