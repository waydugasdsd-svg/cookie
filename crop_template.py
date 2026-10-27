"""Вырезает эталон объекта из скриншота для автоматической разметки.

Запуск:
    python crop_template.py raw/images/screenshot_xxx.png ad_close

Откроется окно: обведите объект мышкой, нажмите ENTER (или SPACE).
Для отмены — клавиша C. Эталон сохранится в templates/<класс>/N.png

Совет: для каждого класса сделайте 2-4 эталона (разный вид объекта).
"""
import os
import sys

import cv2
import numpy as np


def read_image(path):
    # imdecode + fromfile работает и с русскими буквами в путях
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    image_path, class_name = sys.argv[1], sys.argv[2]
    img = read_image(image_path)
    if img is None:
        sys.exit(f"Не удалось открыть {image_path}")

    # окно подгоняем под экран, координаты возвращаем в оригинальном масштабе
    scale = min(1.0, 1400 / img.shape[1], 800 / img.shape[0])
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img

    x, y, w, h = cv2.selectROI("Обведите объект и нажмите ENTER", view, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        sys.exit("Ничего не выбрано.")

    x, y, w, h = (int(round(v / scale)) for v in (x, y, w, h))
    crop = img[y:y + h, x:x + w]

    out_dir = os.path.join("templates", class_name)
    os.makedirs(out_dir, exist_ok=True)
    n = len([f for f in os.listdir(out_dir) if f.endswith(".png")]) + 1
    out_path = os.path.join(out_dir, f"{n}.png")
    ok, buf = cv2.imencode(".png", crop)
    buf.tofile(out_path)
    print(f"Сохранено: {out_path} ({w}x{h})")


if __name__ == "__main__":
    main()
