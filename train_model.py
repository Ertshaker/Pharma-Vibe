from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()

    model = YOLO('yolov8n.pt')  # или 'yolov8s.pt' для меньшей модели
    results = model.train(
        data='data.yaml',
        epochs=45,  # количество эпох
        batch=16,  # размер батча
        imgsz=640,  # размер изображения
        device=0,
        lr0=0.001,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.2,
        weight_decay=0.0005,
        patience=10
    )
