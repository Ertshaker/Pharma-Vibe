from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()

    model = YOLO('yolov8n.pt')  # или 'yolov8s.pt' для меньшей модели
    results = model.train(
        data='data.yaml',
        epochs=50,  # количество эпох
        batch=8,  # размер батча
        imgsz=640,  # размер изображения
        device=0,
        augment=True,
        weight_decay=0.005,
        degrees=0.25,
        scale=0.3,
        perspective=0.0001
    )
