from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS for Flutter Web on Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defined 4 disease classes
CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

@app.get("/")
def read_root():
    return {"status": "Backend service is running."}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    # Read the uploaded image bytes
    image_bytes = await file.read()
    
    # -------------------------------------------------------------
    # TODO: Connect your AI model prediction here
    # Example: prediction_index, confidence = my_model.predict(image_bytes)
    # -------------------------------------------------------------
    
    disease_name = CLASS_NAMES[0]  # Replace with prediction result from your model
    confidence_score = 0.95        # Replace with model confidence (0.0 - 1.0)

    return {
        "class": disease_name,
        "confidence": confidence_score,
        "lat": lat,
        "lng": lng
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)