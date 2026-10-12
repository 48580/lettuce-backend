from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import tensorflow as tf

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

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get model input shape
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Preprocess image
        image = image.resize((width, height))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        predicted_index = int(np.argmax(predictions))
        confidence_score = float(np.max(predictions))
        
        if confidence_score > 1.0:
            confidence_score = 0.95

        disease_name = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else "Other Diseases"

        return {
            "class": disease_name,
            "confidence": round(confidence_score, 4),
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)