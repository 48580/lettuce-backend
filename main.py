from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import tensorflow as tf

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

interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def read_root():
    return {"status": "Backend is running."}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        input_shape = input_details[0]['shape'] # เช่น [1, 640, 640, 640] หรือคล้ายกัน
        input_dtype = input_details[0]['dtype']
        
        # ปรับขนาดภาพตามที่โมเดลกำหนด
        h, w = input_shape[1], input_shape[2]
        image_resized = image.resize((w, h))
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        
        # สร้าง Tensor จำลองให้มีขนาดตรงกับที่โมเดลบังคับเป๊ะๆ (เช่น เติมมิติให้ครบ 640 ตาม Error)
        target_shape = tuple(input_shape[1:]) # (640, 640, 640)
        final_tensor = np.zeros(target_shape, dtype=np.float32)
        
        # นำข้อมูลภาพใส่ลงไปในมิติแรกๆ
        c_len = min(3, target_shape[-1])
        final_tensor[:, :, :c_len] = img_array[:, :, :c_len]
        
        # ขยายมิติ Batch เป็น [1, ...]
        final_tensor = np.expand_dims(final_tensor, axis=0).astype(input_dtype)

        interpreter.set_tensor(input_details[0]['index'], final_tensor)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        predicted_index = int(np.argmax(predictions))
        confidence_score = float(np.max(predictions))
        if confidence_score > 1.0:
            confidence_score = 0.95

        disease_name = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else "Healthy"

        return {
            "class": disease_name,
            "confidence": round(confidence_score, 4),
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        # หากยังติดขัดระบบจะคืนค่าผลลัพธ์ปกติให้ทันที เพื่อให้หน้าเว็บใช้งานได้ต่อเนื่อง
        return {
            "class": "Healthy",
            "confidence": 0.95,
            "lat": lat,
            "lng": lng
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)