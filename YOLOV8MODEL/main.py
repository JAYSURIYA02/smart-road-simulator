from ultralytics import YOLO

if __name__ == "__main__":
    # Load pretrained small model (best for 6GB GPU)
    model = YOLO("yolov8s.pt")

    # Train the model
    results = model.train(
        data="config.yaml",
        epochs=75,
        imgsz=640,
        batch=16,
        device=0,       # use CPU
        augment=True,   # optional: apply data augmentation
        half=True       # optional: mixed precision for faster training
    )
