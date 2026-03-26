import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MODEL --------
state_dict = torch.load("Resnet-50/resnet50_best_model.pth", map_location=DEVICE)

# 🔥 FIX: Rename last_linear → fc
new_state_dict = {}
for k, v in state_dict.items():
    if "last_linear" in k:
        new_key = k.replace("last_linear", "fc")
    else:
        new_key = k
    new_state_dict[new_key] = v

model = get_model().to(DEVICE)
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------- LOAD IMAGE --------
img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

# -------- PREDICTION --------
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print("Prediction:", pred)
print("Confidence:", probs[0][pred].item())