"""
AgriDSS Pro — Model API Server
Deploy this on Render as a Web Service.
Endpoint: POST /predict  →  { "label": "...", "confidence": 0.92, "confidences": [...] }
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

app = FastAPI(title="AgriDSS Model API")

# Allow all origins (so your Streamlit app on any host can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class list ────────────────────────────────────────────────────────────────
CLASSES = [
    'Tomato_Late_blight',
    'Tomato_Septoria_leaf_spot',
    'Tomato_healthy',
    'Tomato_Leaf_Mold',
    'Rice_Bacterial_leaf_blight',
    'Corn_Gray_Leaf_Spot',
    'Corn_Blight',
    'Tomato__Tomato_mosaic_virus',
    'Corn_Healthy',
    'Corn_Common_Rust',
    'Pepper__bell___Bacterial_spot',
    'Rice_Leaf smut',
    'Rice_Brown_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Black_Soil',
    'Cinder_Soil',
    'Laterite_Soil',
    'Peat_Soil',
    'Yellow_Soil',
]

# ── Image transforms (same as training) ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load model once at startup ─────────────────────────────────────────────────
device = torch.device("cpu")   # Render free tier has no GPU

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    # ⚠️  Place your trained weights file (e.g. best_model.pth) in the same
    #     directory and update the filename below.
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "classes": len(CLASSES)}

# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx  = int(np.argmax(probs))
    label    = CLASSES[top_idx]
    conf     = float(probs[top_idx])

    confidences = [
        {"label": CLASSES[i], "confidence": float(probs[i])}
        for i in np.argsort(probs)[::-1]
    ]

    return {
        "label":       label,
        "confidence":  round(conf, 6),
        "confidences": confidences,
    }
