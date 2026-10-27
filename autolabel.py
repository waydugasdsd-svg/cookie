"""Автоматическая разметка данных.

Скрипт сам расставляет рамки на скриншотах и пишет YOLO-разметку (.txt),
так что вручную в X-AnyLabeling остаётся только проверить результат.

Два источника рамок (можно использовать вместе):
  1. Эталоны (template matching): вы вырезаете 2-4 примера каждого объекта
     командой crop_template.py, а скрипт находит такие же объекты на всех
     остальных скриншотах (с разными масштабами).
  2. Уже обученная модель (--model best.pt): она размечает те классы,
     которые знает (например "game"). Это self-training: обучили ->
     разметили новые кадры -> проверили -> обучили заново.

Примеры:
    python autolabel.py                                  # только эталоны
    python autolabel.py --model best.pt                  # эталоны + обученная модель
    python autolabel.py --merge --preview preview        # дополнить старую разметку + картинки для проверки

Картинки берутся из папки screenshorts (её создаёт screener.py), а .txt с разметкой
кладутся рядом с ними — ровно как после экспорта из X-AnyLabeling.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}
COLORS = [(0, 200, 0), (0, 0, 255), (255, 120, 0), (0, 200, 255), (200, 0, 200), (120, 120, 120)]


def read_classes(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    buf.tofile(str(path))


# ---------- геометрия ----------
# Детекция = (class_id, x1, y1, x2, y2, score)

def iou(a, b):
    iw = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ih = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = iw * ih
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thr):
    """Убирает дубли рамок. Работает отдельно внутри каждого класса."""
    keep = []
    for cls in sorted({d[0] for d in dets}):
        items = sorted((d for d in dets if d[0] == cls), key=lambda d: -d[5])
        while items:
            best = items.pop(0)
            keep.append(best)
            items = [d for d in items if iou(best, d) < iou_thr]
    return keep


# ---------- источники рамок ----------

def load_templates(templates_dir, classes):
    """templates/<имя_класса>/*.png -> [(class_id, картинка), ...]"""
    result = []
    root = Path(templates_dir)
    if not root.exists():
        return result
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name not in classes:
            print(f"[!] папка эталонов '{sub.name}' не найдена в class.txt — пропускаю")
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                img = read_image(f)
                if img is not None:
                    result.append((classes.index(sub.name), img))
    return result


def template_detections(img, templates, scales, thr, work_width):
    h, w = img.shape[:2]
    f = min(1.0, work_width / w)  # для скорости ищем на уменьшенной копии
    work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    dets = []
    for cls_id, tmpl in templates:
        for s in scales:
            k = f * s
            t = cv2.resize(tmpl, None, fx=k, fy=k,
                           interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            th, tw = t.shape[:2]
            if th < 6 or tw < 6 or th > work.shape[0] or tw > work.shape[1]:
                continue
            res = cv2.matchTemplate(work, t, cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            ys, xs = np.where(res >= thr)
            if len(ys) == 0:
                continue
            scores = res[ys, xs]
            if len(scores) > 300:  # оставляем лучшие совпадения
                top = np.argpartition(-scores, 300)[:300]
                ys, xs, scores = ys[top], xs[top], scores[top]
            for y, x, sc in zip(ys, xs, scores):
                dets.append((cls_id, x / f, y / f, (x + tw) / f, (y + th) / f, float(sc)))
    return dets


def model_detections(model, img, class_to_id, conf, imgsz):
    r = model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = []
    for b in r.boxes:
        name = model.names[int(b.cls[0])]
        if name not in class_to_id:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append((class_to_id[name], x1, y1, x2, y2, float(b.conf[0])))
    return dets


# ---------- файлы разметки ----------

def read_label_file(path, w, h):
    dets = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        dets.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 2.0))
    return dets


def to_yolo_line(d, w, h):
    c, x1, y1, x2, y2, _ = d
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def draw_preview(img, dets, classes, out_path, max_width=1280):
    vis = img.copy()
    for c, x1, y1, x2, y2, _ in dets:
        col = COLORS[c % len(COLORS)]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, max(2, img.shape[1] // 600))
        cv2.putText(vis, classes[c], (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, img.shape[1] / 1800), col, 2)
    if vis.shape[1] > max_width:
        vis = cv2.resize(vis, None, fx=max_width / vis.shape[1], fy=max_width / vis.shape[1])
    write_image(out_path, vis)


def main():
    ap = argparse.ArgumentParser(description="Автоматическая разметка в формате YOLO")
    ap.add_argument("--images", default="screenshorts", help="папка со скриншотами")
    ap.add_argument("--labels", default=None, help="куда писать .txt (по умолчанию рядом с картинками)")
    ap.add_argument("--classes", default="class.txt")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--model", default=None, help="обученные веса (best.pt) для разметки известных классов")
    ap.add_argument("--model-conf", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.85, help="порог совпадения эталона (0..1); выше = строже")
    ap.add_argument("--scales", default="0.8,0.9,1.0,1.1,1.25", help="масштабы эталона через запятую")
    ap.add_argument("--iou", type=float, default=0.4, help="порог удаления дублей рамок")
    ap.add_argument("--work-width", type=int, default=1280, help="ширина, до которой сжимается кадр для поиска")
    ap.add_argument("--merge", action="store_true", help="сохранить существующую разметку и добавить новые рамки")
    ap.add_argument("--overwrite", action="store_true", help="перезаписать существующую разметку")
    ap.add_argument("--keep-empty", action="store_true", help="писать пустой .txt, если ничего не найдено")
    ap.add_argument("--preview", default=None, help="папка для картинок с нарисованными рамками")
    args = ap.parse_args()

    classes = read_classes(args.classes)
    class_to_id = {n: i for i, n in enumerate(classes)}
    scales = [float(s) for s in args.scales.split(",")]
    templates = load_templates(args.templates, classes)

    model = imgsz = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)
        imgsz = getattr(model, "overrides", {}).get("imgsz", 640)
    if not templates and model is None:
        raise SystemExit("Нет ни эталонов (templates/<класс>/*.png), ни модели (--model). Нечем размечать.")

    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else images_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    counts = {n: 0 for n in classes}
    empty = skipped = done = 0

    for i, path in enumerate(files, 1):
        label_path = labels_dir / (path.stem + ".txt")
        has_label = label_path.exists() and label_path.stat().st_size > 0
        if has_label and not (args.merge or args.overwrite):
            skipped += 1
            continue

        img = read_image(path)
        if img is None:
            print(f"[!] не удалось открыть {path.name}")
            continue
        h, w = img.shape[:2]

        dets = []
        if has_label and args.merge:
            dets += read_label_file(label_path, w, h)  # ручная разметка важнее (score=2.0)
        if model is not None:
            dets += model_detections(model, img, class_to_id, args.model_conf, imgsz)
        if templates:
            dets += template_detections(img, templates, scales, args.thr, args.work_width)
        dets = nms(dets, args.iou)

        if not dets and not args.keep_empty:
            empty += 1
            print(f"[{i}/{len(files)}] {path.name}: ничего не найдено")
            continue

        label_path.write_text("\n".join(to_yolo_line(d, w, h) for d in dets) + ("\n" if dets else ""))
        for d in dets:
            counts[classes[d[0]]] += 1
        done += 1
        print(f"[{i}/{len(files)}] {path.name}: {len(dets)} рамок")

        if args.preview:
            draw_preview(img, dets, classes, Path(args.preview) / (path.stem + ".jpg"))

    print("\n=== Итог ===")
    print(f"размечено кадров: {done}, пропущено (уже размечены): {skipped}, без находок: {empty}")
    for n, c in counts.items():
        print(f"  {n}: {c} рамок")
    if empty:
        print(f"\n[!] В {empty} кадрах ничего не найдено. Если там объекты всё же есть — "
              f"добавьте эталоны или снизьте --thr; если объектов нет, это нормальные 'фоновые' кадры.")
    if args.preview:
        print(f"Проверьте картинки в '{args.preview}/' и поправьте ошибки в X-AnyLabeling (или удалите плохие .txt).")


if __name__ == "__main__":
    main()
