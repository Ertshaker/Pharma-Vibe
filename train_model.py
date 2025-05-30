from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()

    model = YOLO('yolov8n.pt')
    results = model.train(
        data='data.yaml',
        epochs=45,
        batch=16,
        imgsz=640,
        device=0,
        lr0=0.001,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.2,
        weight_decay=0.0005,
        patience=10
    )
