"""Раскладывает размеченные данные по структуре из задания и обновляет data.yaml.

Берёт картинки и .txt из папки screenshorts (картинка + одноимённый .txt рядом)
и делает:
    images/train, images/val, labels/train, labels/val    (80% / 20%)
    data.yaml                                             (классы из class.txt)

Если images/ и labels/ уже есть (старый датасет), запустите с --clean:
старые пары картинка+разметка сначала добавятся в общий набор, потом всё
пересоберётся заново, так что ничего не потеряется.

Запуск:
    python split_dataset.py
    python split_dataset.py --clean
"""
import argparse
import random
import shutil
import tempfile
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg"}


def collect_pairs(img_dir, lbl_dir):
    """{имя_без_расширения: (картинка, разметка)} для картинок, у которых есть .txt."""
    pairs = {}
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMG_EXT:
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs[img.stem] = (img, lbl)
    return pairs


def count_classes(label_files, n_classes):
    counts = [0] * n_classes
    for f in label_files:
        for line in f.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="screenshorts", help="папка с картинками и .txt разметкой")
    ap.add_argument("--dst", default=".", help="куда класть images/ labels/ data.yaml")
    ap.add_argument("--classes", default="class.txt")
    ap.add_argument("--val", type=float, default=0.2, help="доля валидации")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="пересобрать images/ и labels/ (старые данные сохраняются)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    classes = [l.strip() for l in Path(args.classes).read_text(encoding="utf-8").splitlines() if l.strip()]

    pairs = collect_pairs(src, src)
    no_label = [p.name for p in src.iterdir() if p.suffix.lower() in IMG_EXT and p.stem not in pairs] if src.exists() else []
    if no_label:
        print(f"[!] {len(no_label)} картинок без разметки пропущены")

    targets = [dst / a / b for a in ("images", "labels") for b in ("train", "val")]
    exists = any(t.exists() and any(t.iterdir()) for t in targets)
    tmp = None
    if exists:
        if not args.clean:
            raise SystemExit("images/ и labels/ уже заполнены. Запустите с --clean: старые данные "
                             "добавятся к новым и всё будет разложено заново.")
        # старые пары переносим во временную папку, чтобы они пережили удаление
        tmp = Path(tempfile.mkdtemp())
        old = 0
        for part in ("train", "val"):
            for stem, (img, lbl) in collect_pairs(dst / "images" / part, dst / "labels" / part).items():
                if stem not in pairs:
                    shutil.copy2(img, tmp / img.name)
                    shutil.copy2(lbl, tmp / lbl.name)
                    old += 1
        for t in targets:
            shutil.rmtree(t, ignore_errors=True)
        merged = collect_pairs(tmp, tmp)  # старые данные
        merged.update(pairs)              # новые перекрывают старые с тем же именем
        pairs = merged
        print(f"добавлено из старого датасета: {old}")

    items = list(pairs.values())
    if len(items) < 2:
        raise SystemExit("Мало размеченных картинок (нужно минимум 2).")

    random.Random(args.seed).shuffle(items)
    n_val = min(len(items) - 1, max(1, round(len(items) * args.val)))
    split = {"val": items[:n_val], "train": items[n_val:]}

    for name, part in split.items():
        (dst / "images" / name).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / name).mkdir(parents=True, exist_ok=True)
        for img, lbl in part:
            shutil.copy2(img, dst / "images" / name / img.name)
            shutil.copy2(lbl, dst / "labels" / name / lbl.name)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    names = ", ".join(f"'{c}'" for c in classes)
    (dst / "data.yaml").write_text(
        f"path: ./\ntrain: images/train\nval: images/val\n\nnc: {len(classes)}\nnames: [{names}]\n",
        encoding="utf-8",
    )

    print(f"train: {len(split['train'])} картинок, val: {len(split['val'])} картинок")
    for name, part in split.items():
        counts = count_classes(sorted((dst / "labels" / name).glob("*.txt")), len(classes))
        print(f"  {name}: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
        for c, n in zip(classes, counts):
            if n == 0:
                print(f"  [!] в {name} нет ни одного объекта класса '{c}'")
            elif n < 10 and name == "train":
                print(f"  [!] класса '{c}' в train всего {n} — маловато, лучше 30+")
    print("data.yaml обновлён.")


if __name__ == "__main__":
    main()
