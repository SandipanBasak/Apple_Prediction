from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

# -------------------- APP CONFIG --------------------
app = FastAPI(title="Apple Disease Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use ["http://localhost:5173"] for stricter rule
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "apple_resnet50_model_full.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- CLASS NAMES --------------------
CLASS_NAMES = {
    0: "Apple Scab",
    1: "Black Rot",
    2: "Cedar Apple Rust",
    3: "Healthy"
}

# -------------------- CURE & PREVENTION DATA --------------------
DISEASE_INFO = {
    "Apple Scab": {
        "cure_map": [
            "Use resistant apple varieties like 'Liberty' or 'Enterprise'.",
            "Apply fungicides such as Captan or Mancozeb at early leaf stage.",
            "Remove and destroy fallen infected leaves to reduce spore load."
        ],
        "prevention_map": [
            "Ensure proper air circulation through pruning.",
            "Avoid overhead irrigation to minimize leaf wetness.",
            "Follow a preventive fungicide spray schedule during wet seasons."
        ]
    },
    "Black Rot": {
        "cure_map": [
            "Prune and burn infected twigs, cankers, and mummified fruits.",
            "Apply fungicides such as Captan or Thiophanate-methyl during fruit development.",
            "Remove nearby infected trees or dead wood where the fungus may overwinter."
        ],
        "prevention_map": [
            "Keep the orchard floor clean by removing fallen leaves and fruits.",
            "Disinfect pruning tools after use.",
            "Use disease-free planting material."
        ]
    },
    "Cedar Apple Rust": {
        "cure_map": [
            "Remove nearby red cedar or juniper trees that act as alternate hosts.",
            "Apply fungicides (e.g., Mancozeb, Myclobutanil) at the pink bud stage and repeat as needed.",
            "Prune infected leaves and twigs promptly."
        ],
        "prevention_map": [
            "Plant resistant apple varieties such as 'Freedom' or 'Redfree'.",
            "Maintain proper spacing between trees for good air flow.",
            "Regularly inspect for early signs of rust galls on cedar or apple leaves."
        ]
    },
    "Healthy": {
        "cure_map": ["No treatment required — leaf is healthy."],
        "prevention_map": [
            "Maintain regular fertilization and irrigation schedule.",
            "Inspect orchard regularly for early signs of disease.",
            "Ensure balanced nutrient management and pest monitoring."
        ]
    }
}

# -------------------- MODEL LOADING --------------------
def load_model():
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully: {type(model)}")
    return model

model = load_model()

# -------------------- IMAGE PREPROCESSING --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- ROUTES --------------------
@app.get("/")
def home():
    return {"message": "✅ Apple Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform and predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[preds.item()]

        # Get recommendations
        disease_info = DISEASE_INFO.get(predicted_class, {"cure_map": [], "prevention_map": []})

        return JSONResponse({
            "prediction": predicted_class,
            "cure_recommendations": disease_info["cure_map"],
            "prevention_tips": disease_info["prevention_map"]
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
