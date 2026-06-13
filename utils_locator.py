import os
import glob

def get_latest_output():
    base_dir = "runs/detect"
    pattern = os.path.join(base_dir, "train*")
    train_dirs = glob.glob(pattern)
    if not train_dirs:
        return None, None

    latest_dir = max(train_dirs, key=os.path.getmtime)
    video_path = os.path.join(latest_dir, "output_with_plates.mp4")
    txt_path = os.path.join(latest_dir, "final_detected_plates.txt")

    video_path = video_path.replace("\\", "/") if os.path.exists(video_path) else None
    txt_path = txt_path.replace("\\", "/") if os.path.exists(txt_path) else None

    return video_path, txt_path
