import cv2
import easyocr
from ultralytics import YOLO

# Инициализация модели YOLO и EasyOCR
model = YOLO("runs/detect/train3/weights/best.pt")
reader = easyocr.Reader(["ru"])

img = cv2.imread("ocr_test/img_2.png")

results = model(img)

for i, box in enumerate(results[0].boxes):
    bbox = box.xyxy[0].cpu().numpy()
    cls_id = int(box.cls)
    conf = float(box.conf)

    x1, y1, x2, y2 = map(int, bbox)
    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(f"ocr_test/object_{i}.jpg", cropped)

    text = reader.readtext(cropped)

    # Вывод информации
    print(f"Объект {i}:")
    print(f"  Класс: {cls_id} (Уверенность: {conf:.2f})")
    print(f"  Координаты: [{x1}, {y1}, {x2}, {y2}]")
    print(f"  Распознанный текст:", text)
    print("-" * 50)

# Визуализация результатов (если нужно)
results_plotted = results[0].plot()  # Рисует bbox и подписи на изображении
cv2.imwrite("ocr_test/detected_objects.jpg", results_plotted)
