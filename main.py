from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image

app = FastAPI()

# Enable CORS for Flutter Web on Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defined 4 disease classes (Must match your AI model's training order)
CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

@app.get("/")
def read_root():
    return {"status": "Lettuce Disease Detection Backend is running."}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # -------------------------------------------------------------
        # TODO: Insert your actual AI model prediction code here
        # Example: 
        #   processed_img = preprocess(image)
        #   prediction_index, confidence = my_model.predict(processed_img)
        # -------------------------------------------------------------
        
        # Mock prediction result (Replace this with your model output)
        predicted_index = 0       # 0 = Healthy, 1 = Powdery Mildew, etc.
        confidence_score = 0.96   # Confidence score (0.0 - 1.0)

        # Ensure index is within range
        if predicted_index >= len(CLASS_NAMES):
            predicted_index = 3

        disease_name = CLASS_NAMES[predicted_index]

        return {
            "class": disease_name,
            "confidence": confidence_score,
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)