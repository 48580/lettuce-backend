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

# รายชื่อคลาสทั้ง 4 ตามลำดับ
CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

# โหลดโมเดล TFLite
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
        
        # ดึงขนาด input ที่โมเดลต้องการจากไฟล์ tflite โดยตรง (เช่น [1, 640, 640, 3] หรือ [1, 224, 224, 3])
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        height = input_shape[1]
        width = input_shape[2]
        
        # ปรับขนาดภาพให้เท่ากับที่โมเดลกำหนด
        image_resized = image.resize((width, height))
        
        # แปลงข้อมูลรูปภาพให้เป็นประเภทเดียวกับที่โมเดลคาดหวัง
        if input_dtype == np.float32:
            img_array = np.array(image_resized, dtype=np.float32) / 255.0
        else:
            # สำหรับโมเดล int8/uint8
            img_array = np.array(image_resized, dtype=input_dtype)
            
        # เติมมิติ Batch เข้าไปให้เป็น [1, height, width, channels]
        img_array = np.expand_dims(img_array, axis=0)

        # ส่งข้อมูลเข้าโมเดล TFLite เพื่อทำนายผลจริง
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        # หาคลาสที่มีค่าสูงสุด
        predicted_index = int(np.argmax(predictions))
        
        # แปลงค่า Output เป็นความน่าจะเป็น (Softmax ถ้ายังไม่ได้แปลง)
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        confidence_score = float(probabilities[predicted_index])

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