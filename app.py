import gradio as gr
import cv2
import easyocr
from ultralytics import YOLO
from PIL import Image
import numpy as np

css = """
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300..700&display=swap');

body {
    background-image:url("prikol.jpg");
    font-family: "Comfortaa", sans-serif;
    color: white;
}

footer > * {
    color: black;
}

.gradio-container {
    background-color: #7cfc00;
    background-image: linear-gradient(315deg, #7cfc00 0%, #0bda51 74%);
    border-radius: 48px;
    padding: 2rem;
    min-width: 500px;
    max-width: 850px;
    margin: 2rem auto;
    box-shadow: 0 8px 20px rgba(0, 128, 128, 0.3);
}

.header > * {
    font-family: "Comfortaa", sans-serif;
    font-weight: 600;
    font-size: 48px;
    text-align: center;
}

/* Кнопки */
.gr-button {
    background-color: white!important;
    color: white !important;
    font-weight: bold;
    border-radius: 6px !important;
    padding: 10px 18px !important;
    transition: background 0.2s ease;
    border: none !important;
}

.analyze_button,  {
    background-color: white
    color: white;
    font-weight: bold;
    border-radius: 18px;
    padding: 10px 18px;
    transition: background 0.2s ease;
    border: none;
}

.clear_button {
    background-color: red;
    color: white !important;
    font-weight: bold;
    border-radius: 18px !important;
    padding: 10px 18px !important;
    transition: background 0.2s ease;
    border: none !important;
}

.gr-button:hover {
    background-color: #00838f !important;
}

.output_field {
    border: 2px solid #b2ebf2;
    border-radius: 18px;
    background: white;
    border-radius: 18px;
    font-size: 24px;
    padding: 10px;
    color: #004d5a;
}

.image_upload {
    border: 2px dashed #4dd0e1;
    background: white;
    border-radius: 18px;
    padding: 1rem;
    transition: background 0.3s ease;
    width: 400px;
    height: 400px;
    object-fit: contain;
}

.image_upload:hover {
    background: #b2ebf2;
}

/* Изображения (вывод) */
.gr-image {
    border: 1px solid #b2ebf2;
    background: white;
    border-radius: 18px;
    padding: 5px;
    width: 400px;
    height: 400px;
    object-fit: contain;
}

.gradio-container, .gr-button, .gr-textbox, .gr-image {
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    0%   { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}
"""

class_match = {
    0: "qr",
    1: "barcode",
    2: "exp_date",
    4: "name"
}

model = YOLO("../runs/detect/legendary_final/weights/best.pt")
reader = easyocr.Reader(["ru"])


class FoundedObjects:
    objects = {
        4: None,
        2: None,
        0: None,
        1: None
    }

    @staticmethod
    def clear():
        FoundedObjects.objects = {
            4: None,
            2: None,
            0: None,
            1: None
        }
        return [None, None, None, None, None]


def process_images(image):
    if image is None:
        return [
            FoundedObjects.objects[4] if FoundedObjects.objects[4] is not None else "Не найдено",
            FoundedObjects.objects[2] if FoundedObjects.objects[2] is not None else "Не найдено",
            FoundedObjects.objects[1] if FoundedObjects.objects[1] is not None else gr.Image("not_found.png"),
            FoundedObjects.objects[0] if FoundedObjects.objects[0] is not None else gr.Image("not_found.png"),
            None
        ]

    # порог уверенности??? можно иметь можно убрать ваще, я пока оставлю
    conf_threshold = 0.4
    text = ''
    for i, box in enumerate(model(image, conf=conf_threshold)[0].boxes):
        bbox = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls)
        conf = float(box.conf)

        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]

        # cls_id = 3 должен сдохнуть в зоне, паразитический класс
        if cls_id == 3:
            continue

        if FoundedObjects.objects[cls_id] is not None:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        if cls_id == 2 or cls_id == 4:
            text = "".join(reader.readtext(cropped, detail=0))
            FoundedObjects.objects[cls_id] = text

        if cls_id == 0 or cls_id == 1:
            FoundedObjects.objects[cls_id] = cropped

        print(f"Объект {i + 1}:\n"
              f"\tКласс: {class_match[cls_id]}\n"
              f"\tУверенность: {conf:.2f}\n"
              f"\tx1, y1: {x1}, {y1}\n"
              f"\tx2, y2: {x2}, {y2}\n"
              f"\tРаспознанный текст:", text)

        print("=" * 75)

    return [
        FoundedObjects.objects[4] if FoundedObjects.objects[4] is not None else "Не найдено",
        FoundedObjects.objects[2] if FoundedObjects.objects[2] is not None else "Не найдено",
        FoundedObjects.objects[1] if FoundedObjects.objects[1] is not None else gr.Image("not_found.png"),
        FoundedObjects.objects[0] if FoundedObjects.objects[0] is not None else gr.Image("not_found.png"),
        None
    ]


with gr.Blocks(css=css) as demo:
    gr.Markdown("Pharma-Vibe", elem_classes="header")
    with gr.Row():
        image_input = gr.Image(
            type="numpy",
            label="📷 Загрузите изображение",
            elem_classes=["image_upload"]
        )

    with gr.Row():
        btn_analyze = gr.Button("🔍 Анализировать", elem_id="analyze_button")
        btn_clear = gr.Button("🗑 Очистить всё", elem_id="clear_button")

    with gr.Row():
        out_name = gr.Textbox(label="📝 Наименование",
                              interactive=False,
                              elem_classes=["output_field"])

        out_expiry = gr.Textbox(label="⏳ Срок годности",
                                interactive=False,
                                elem_classes=["output_field"])

    with gr.Row():
        out_barcode = gr.Image(label="📎 Штрих‑код",
                               interactive=False,
                               type="numpy", elem_classes=["gr-image"])

        out_qr = gr.Image(label="📱 QR‑код",
                          interactive=False,
                          type="numpy", elem_classes=["gr-image"])

    btn_analyze.click(fn=process_images,
                      inputs=[image_input],
                      outputs=[out_name, out_expiry, out_barcode, out_qr, image_input])

    btn_clear.click(fn=FoundedObjects.clear, outputs=[out_name, out_expiry, out_barcode, out_qr, image_input])

if __name__ == "__main__":
    demo.launch()
