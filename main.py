from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
import sqlite3
import cv2
import numpy as np

# 1. ประกาศตัวแปร app สำหรับ FastAPI (อยู่ตรงนี้ครับ)
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware  # นำเข้าตัวจัดการ CORS

# อนุญาตให้ Flutter และ Web จากภายนอกส่งรูปเข้ามาวิเคราะห์ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2. โหลดโมเดล YOLOv8
model = YOLO("best_int8.tflite")

# 3. สร้างฐานข้อมูล SQLite บันทึกผล
def init_db():
    conn = sqlite3.connect("farm_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            zone_x INTEGER,
            zone_y INTEGER,
            disease_detected TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 4. API Endpoint สำหรับรับรูปภาพและพิกัด
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    lat: float = Form(...), 
    lng: float = Form(...)
):
    # อ่านไฟล์รูปภาพจากแอป
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ให้ AI ประมวลผลหาโรคผักกาด
    results = model(img)
    detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(model.names[int(c)])

    disease_str = ", ".join(detected_classes) if detected_classes else "Healthy"

    # คำนวณตารางกริด 3x3 เมตร (Smart Grid Mapping)
    zone_x = int(lat * 100000) % 3
    zone_y = int(lng * 100000) % 3

    # บันทึกลงฐานข้อมูล SQLite
    conn = sqlite3.connect("farm_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO detections (latitude, longitude, zone_x, zone_y, disease_detected) VALUES (?, ?, ?, ?, ?)",
        (lat, lng, zone_x, zone_y, disease_str)
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "disease": disease_str,
        "location": {"lat": lat, "lng": lng},
        "zone": f"Zone [{zone_x},{zone_y}]"
    }