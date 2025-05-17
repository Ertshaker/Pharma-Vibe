import cv2
import easyocr
from ultralytics import YOLO
import os

class_match = {
    0: "QR",
    1: "barcode (Штрихкод)",
    2: "date (Дата)",
    3: "expire_date (Срок годности)",
    4: "name (Наименование)"
}

# Удаление результатов с прошлого теста
if os.path.exists("detected_objects.jpg"):
    os.remove("detected_objects.jpg")
for i in range(10):
    if os.path.exists(os.path.join(f"object_{i}.jpg")):
        os.remove(f"object_{i}.jpg")
    else:
        continue

# Модели находятся в runs/detect.
# newbie - самое первое обучение модели.
# legendary_final - гигачадская легенда, которую может победить только увеличенный датасет.
# train(число) - промежуточные варианты. Тотальный хаос.
model = YOLO("../runs/detect/legendary_final/weights/best.pt")
reader = easyocr.Reader(["ru", "en"])
# изображения для теста в non_seen_images
img = cv2.imread("non_seen_images/5.jpg")

results = model(img)

for i, box in enumerate(results[0].boxes):
    bbox = box.xyxy[0].cpu().numpy()
    cls_id = int(box.cls)
    conf = float(box.conf)

    x1, y1, x2, y2 = map(int, bbox)
    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(f"object_{i+1}.jpg", cropped)

    text = reader.readtext(cropped, detail=0)

    print(f"Объект {i+1}:\n"
          f"\tКласс: {class_match[cls_id]}\n"
          f"\tУверенность: {conf:.2f}\n"
          f"\tx1, y1: {x1}, {y1}\n"
          f"\tx2, y2: {x2}, {y2}\n"
          f"\tРаспознанный текст:", "".join(text))
    print("=" * 75)

results_plotted = results[0].plot()
cv2.imwrite("detected_objects.jpg", results_plotted)
