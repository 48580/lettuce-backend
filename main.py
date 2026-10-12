from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image

# สมมติถ้าใช้ TensorFlow/Keras:
# import tensorflow as tf
# model = tf.keras.models.load_model('your_model_file.h5')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # -------------------------------------------------------------
        # 💡 ส่วนที่ต้องเปลี่ยน: นำภาพเข้าโมเดล AI ของคุณจริง ๆ ตรงนี้
        # -------------------------------------------------------------
        
        # 1. ปรับขนาดภาพให้ตรงกับที่โมเดลต้องการ (เช่น 224x224 พิกเซล)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. สั่งให้โมเดลทำนายผล (ตัวอย่างโครงสร้าง TensorFlow)
        # predictions = model.predict(img_array)
        # predicted_index = int(np.argmax(predictions[0]))
        # confidence_score = float(np.max(predictions[0]))
        
        # (ชั่วคราว: หากยังไม่ได้ต่อโมเดล ให้บอกผมได้ครับ)
        predicted_index = 0      
        confidence_score = 0.95  

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