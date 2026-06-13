from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(data='data.yaml', epochs=20, imgsz=640,save=True)
