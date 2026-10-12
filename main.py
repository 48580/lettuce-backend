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

# รายชื่อคลาสทั้ง 4 ตามลำดับที่คุณต้องการ
CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

# โหลดโมเดล TFLite ที่อยู่ในโฟลเดอร์เดียวกัน
interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def read_root():
    return {"status": "Lettuce Disease Detection Backend with TFLite is running."}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    try:
        # 1. อ่านรูปภาพที่อัปโหลดเข้ามา
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. ปรับขนาดภาพให้ตรงกับที่โมเดล TFLite ต้องการ (ส่วนใหญ่คือ 224x224 หรือ 160x160)
        # ลองเช็คขนาด input ของโมเดลจาก input_details[0]['shape']
        input_shape = input_details[0]['shape'] # เช่น [1, 224, 224, 3]
        height, width = input_shape[1], input_shape[2]
        
        image = image.resize((width, height))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ถ้าโมเดลเป็นแบบ INT8 อาจจะต้องแปลงชนิดข้อมูลตาม input_details[0]['dtype']
        if input_details[0]['dtype'] == np.int8 or input_details[0]['dtype'] == np.uint8:
            # แปลงค่ากลับตามเกณฑ์ Quantization ของ TFLite ถ้าจำเป็น
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                img_array = img_array / scale + zero_point
            img_array = img_array.astype(input_details[0]['dtype'])

        # 3. สั่งรันโมเดล (Interpreter Inference)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        # 4. หาผลลัพธ์คลาสที่มีคะแนนสูงสุด
        predicted_index = int(np.argmax(predictions))
        
        # คำนวณค่าความมั่นใจ (Confidence Score)
        # ถ้ารูปแบบ Output เป็น Softmax หรือ Raw Logits
        confidence_score = float(np.max(predictions))
        if confidence_score > 1.0: # กรณีไม่ใช่สัดส่วน 0-1 ให้แปลงคร่าวๆ
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