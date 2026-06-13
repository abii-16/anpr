from ultralytics import YOLO

# Load your trained model
model = YOLO('best.pt')

# Run prediction on a video
model.predict(
    source='vid2.mp4',   # path to video
    save=True,               # save output video with predictions
    imgsz=320,               # inference image size
    conf=0.2                 # confidence threshold
)
