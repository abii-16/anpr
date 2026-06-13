from flask import Flask, render_template, request, send_from_directory
import os
import uuid
import shutil
import subprocess
import time
from utils_locator import get_latest_output
import mysql.connector

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
VIDEO_FOLDER = os.path.join(STATIC_FOLDER, "videos")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",              # Change this to your MySQL username
    password="abi123",  # Change this to your MySQL password
    database="anpr"
)
cursor = db.cursor()

# Create tables if they don't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS videos (
    id VARCHAR(100) PRIMARY KEY,
    original_filename VARCHAR(255),
    processed_video_path TEXT,
    detected_text_path TEXT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS plates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id VARCHAR(100),
    plate_number VARCHAR(200),
    plate_time VARCHAR(20),
    FOREIGN KEY (video_id) REFERENCES videos(id)
)
""")
db.commit()

@app.route("/", methods=["GET", "POST"])
def index():
    video_download = None
    text_download = None
    detected_plates = []

    if request.method == "POST":
        uploaded_file = request.files["video"]
        if uploaded_file and uploaded_file.filename.endswith(".mp4"):
            video_id = str(uuid.uuid4())
            saved_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
            uploaded_file.save(saved_path)

            # Run detection script
            subprocess.run([
                "python", "predict_modified.py",
                f"model=best.pt",
                f"source={saved_path}"
            ])

            video_path, txt_path = get_latest_output()
            print(f"📦 Video found: {video_path}")
            print(f"📄 Text file found: {txt_path}")

            if video_path and txt_path:
                out_video_filename = f"{video_id}_output.mp4"
                out_txt_filename = f"{video_id}_output.txt"

                final_video_path = os.path.join(VIDEO_FOLDER, out_video_filename)
                final_txt_path = os.path.join(VIDEO_FOLDER, out_txt_filename)

                shutil.copy(video_path, final_video_path)
                shutil.copy(txt_path, final_txt_path)

                time.sleep(0.5)

                if os.path.exists(final_video_path):
                    video_download = out_video_filename
                if os.path.exists(final_txt_path):
                    text_download = out_txt_filename

                # Read detected plates and timestamps from file
                with open(final_txt_path) as f:
                    lines = f.read().splitlines()
                    detected_plates = []
                    for line in lines:
                        if " at " in line:
                            plate, timestamp = line.split(" at ")
                            detected_plates.append((plate.strip(), timestamp.strip()))

                # Save video metadata in DB
                cursor.execute("""
                    INSERT INTO videos (id, original_filename, processed_video_path, detected_text_path)
                    VALUES (%s, %s, %s, %s)
                """, (video_id, uploaded_file.filename, final_video_path, final_txt_path))

                # Save each plate with timestamp in DB
                for plate, timestamp in detected_plates:
                    cursor.execute("""
                        INSERT INTO plates (video_id, plate_number, plate_time)
                        VALUES (%s, %s, %s)
                    """, (video_id, plate, timestamp))
                db.commit()

    return render_template("index.html",
                           plates=detected_plates,
                           video_download=video_download,
                           text_download=text_download)


@app.route("/videos/<filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
