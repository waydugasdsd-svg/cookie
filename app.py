"""Кликер-бот для Cookie Clicker на YOLOv8.

Что делает (по приоритету):
  1. ad_close — закрывает рекламу (клик по крестику + проверка, что она исчезла);
  2. bonus    — кликает по всем найденным бонусам (они быстро исчезают);
  3. upgrade  — покупает доступное улучшение (не чаще раза в UPGRADE_COOLDOWN);
  4. game     — кликает по печеньке.

Путь к модели ищется автоматически: сначала --model, затем runs/detect/*/weights/best.pt
в текущей папке, затем папки cookie_detecror*, затем старый путь из практической.

Аварийная остановка: мышь в угол экрана (pyautogui.FAILSAFE) или Ctrl+C.
На macOS терминалу нужны разрешения «Запись экрана» и «Универсальный доступ».

Запуск:
    python app.py
    python app.py --model runs/detect/cookie_detecror/weights/best.pt --conf 0.35
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01   # pyautogui по умолчанию спит 0.1 с после КАЖДОГО клика — убираем

# ---------- параметры поведения ----------
COOKIE_CLICKS = 30       # кликов по печеньке за один проход
COOKIE_DELAY = 0.01      # пауза между кликами по печеньке, с
BONUS_DELAY = 0.01       # пауза между кликами по разным бонусам, с
UPGRADE_COOLDOWN = 3.0   # как часто можно покупать улучшение, с (можно менять через --upgrade-cooldown)
UPGRADE_ROW_HEIGHT = 45  # примерная высота одной строки магазина, экранных px (--row-height)
UPGRADE_ROW_DELAY = 0.15 # пауза между кликами по соседним строкам магазина, с
UPGRADE_HOVER = 0.05     # сколько курсор «висит» над кнопкой перед кликом, с
UPGRADE_HOLD = 0.05      # сколько удерживаем кнопку мыши на улучшении, с
AD_MAX_TRIES = 5         # максимум попыток закрыть одну рекламу
AD_RECHECK_DELAY = 0.6   # пауза перед повторной проверкой после клика, с
CLASS_PRIORITY = ('ad_close', 'bonus', 'upgrade', 'game')

OLD_MODEL_PATH = '/Users/coteunik/Desktop/python_learn/cookie_clicker_data/yolo/runs/detect/cookie_detecror/weights/best.pt'


def find_model_path(cli_value):
    """Ищем веса модели: аргумент CLI -> runs/detect/* -> старый путь из практической."""
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value))
    for base in (Path('.'), Path(__file__).parent):
        candidates += sorted(
            (base / 'runs' / 'detect').glob('*/weights/best.pt'),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        # train.py сохраняет runs прямо рядом, но иногда в cookie_detecror*
        candidates += sorted(base.glob('cookie_detecror*/weights/best.pt'))
    candidates.append(Path(OLD_MODEL_PATH))

    for c in candidates:
        if c.is_file():
            return c
    searched = '\n  '.join(str(c) for c in candidates)
    sys.exit(f'Не найдена обученная модель. Искал:\n  {searched}\n'
             f'Сначала обучите её: python train.py (см. README), '
             f'либо укажите путь через --model.')


class CookieBot:
    def __init__(self, model_path, conf, upgrade_cooldown, row_height=UPGRADE_ROW_HEIGHT):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.upgrade_cooldown = upgrade_cooldown
        self.row_height = row_height
        # Модель понимает только классы из class.txt — лишние просто игнорируем
        known = set(self.model.names.values()) if hasattr(self.model.names, 'values') \
            else set(self.model.names)
        self.classes = [c for c in CLASS_PRIORITY if c in known]
        missing = [c for c in CLASS_PRIORITY if c not in known]
        if missing:
            print(f'[!] В модели нет классов {missing} — эти действия будут пропущены. '
                  f'Перепроверьте class.txt и дообучите модель.')
        # retina-экраны: пиксели скриншота != пиксели мыши
        self.scale = pyautogui.screenshot().width / pyautogui.size().width
        self.last_upgrade_time = 0.0
        self.stats = {c: 0 for c in CLASS_PRIORITY}

    # ---------- детекция ----------

    def detect(self):
        """Один скриншот -> {класс: [(уверенность, x, y, w, h)]} в экранных координатах.

        x, y — центр рамки; w, h — её ширина и высота (нужны, чтобы «пробежаться»
        кликами по всему списку улучшений, а не бить в одну точку центра)."""
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        results = self.model(frame, conf=self.conf, verbose=False)

        found = {c: [] for c in CLASS_PRIORITY}
        for result in results:
            for box in result.boxes:
                name = self.model.names[int(box.cls[0])]
                if name not in found:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                found[name].append((conf,
                                    (x1 + x2) / 2 / self.scale,
                                    (y1 + y2) / 2 / self.scale,
                                    (x2 - x1) / self.scale,
                                    (y2 - y1) / self.scale))
        return found

    # ---------- пункты 3-5 ----------

    def close_ads(self, found):
        """Пункт 3: закрыть рекламу. Кликаем и проверяем, что она исчезла."""
        for attempt in range(1, AD_MAX_TRIES + 1):
            best = max(found['ad_close'], key=lambda d: d[0])
            conf, x, y = best[:3]
            print(f'Реклама (уверенность {conf:.2f}), попытка {attempt}: клик ({x:.0f}, {y:.0f})')
            pyautogui.click(x, y)
            self.stats['ad_close'] += 1
            time.sleep(AD_RECHECK_DELAY)

            # проверяем, исчезла ли реклама после клика
            found = self.detect()
            if not found['ad_close']:
                print('Реклама закрыта.')
                return found
        print('[!] Реклама не закрылась за отведённые попытки — пропускаю этот кадр.')
        return found

    def click_bonuses(self, found):
        """Пункт 5: кликнуть по всем бонусам снизу вверх (сортируем по y, большие y = ниже)."""
        for conf, x, y, *_ in sorted(found['bonus'], key=lambda d: -d[2]):
            print(f'Бонус (уверенность {conf:.2f}): клик ({x:.0f}, {y:.0f})')
            pyautogui.click(x, y)
            self.stats['bonus'] += 1
            time.sleep(BONUS_DELAY)
        return bool(found['bonus'])

    def buy_upgrade(self, found, now):
        """Пункт 4: купить улучшение.

        Модель размечает классом 'upgrade' ВЕСЬ список магазина одной рамкой,
        поэтому клик в её центр всегда попадает в один и тот же ряд (фабрику).
        Делим рамку на строки и проходим кликами снизу вверх: в магазине внизу
        самые дешёвые здания — их мы почти всегда можем купить, а клик по
        недоступному зданию игра просто игнорирует.

        Возвращает (пробовали_купить, свежий_кадр_детекции)."""
        if not found['upgrade'] or now - self.last_upgrade_time < self.upgrade_cooldown:
            return False, found
        conf, x, y, w, h = max(found['upgrade'], key=lambda d: d[0])
        rows = max(1, round(h / self.row_height))
        print(f'Улучшение (уверенность {conf:.2f}): рамка {w:.0f}x{h:.0f}, '
              f'прохожусь по {rows} строкам снизу вверх')
        top = y - h / 2
        row_h = h / rows
        for i in range(rows):
            # i=0 — нижняя строка (самое дешёвое), дальше вверх по списку
            ry = top + h - (i + 0.5) * row_h
            # плавное наведение + удержанный клик: мгновенный «телепорт» курсора
            # и щелчок на 10 мс игра часто не регистрирует (после pyautogui.PAUSE=0.01)
            pyautogui.moveTo(x, ry, duration=UPGRADE_HOVER)
            pyautogui.mouseDown()
            time.sleep(UPGRADE_HOLD)
            pyautogui.mouseUp()
            self.stats['upgrade'] += 1
            time.sleep(UPGRADE_ROW_DELAY)
        self.last_upgrade_time = time.time()
        return True, self.detect()

    def click_cookie(self, found):
        """Клик по печеньке."""
        if not found['game']:
            return False
        conf, x, y = max(found['game'], key=lambda d: d[0])[:3]
        for _ in range(COOKIE_CLICKS):
            pyautogui.click(x, y)
            time.sleep(COOKIE_DELAY)
        self.stats['game'] += 1
        return True

    # ---------- главный цикл ----------

    def step(self):
        """Один проход: скриншот -> приоритетные действия. True = что-то сделали."""
        found = self.detect()

        # Пункт 3: реклама перекрывает игру — самое важное
        if found['ad_close']:
            found = self.close_ads(found)

        clicked = False

        # Пункт 5: бонусы исчезают быстро — сразу после рекламы
        if self.click_bonuses(found):
            clicked = True

        # Пункт 4: улучшения (магазин в стороне, спешить некуда)
        upgraded, found = self.buy_upgrade(found, time.time())
        if upgraded:
            clicked = True

        # Печенька — фоновый режим
        if self.click_cookie(found):
            clicked = True
        else:
            print('Печеньки на экране нет.')
        return clicked

    def run(self):
        print(f'Бот запущен. Классы: {self.classes}. Стоп: мышь в угол экрана или Ctrl+C.\n')
        try:
            while True:
                acted = self.step()
                time.sleep(0.1 if acted else 0.05)
        except KeyboardInterrupt:
            pass
        finally:
            total = sum(self.stats.values())
            print(f'\nОстановлено. Итого кликов по классам: '
                  f'{", ".join(f"{c}={n}" for c, n in self.stats.items())} (всего {total}).')


def main():
    ap = argparse.ArgumentParser(description='Кликер-бот для Cookie Clicker на YOLOv8')
    ap.add_argument('--model', default=None, help='путь к best.pt (по умолчанию ищется автоматически)')
    ap.add_argument('--conf', type=float, default=0.4, help='порог уверенности детекции (0..1)')
    ap.add_argument('--upgrade-cooldown', type=float, default=UPGRADE_COOLDOWN,
                    help='минимальная пауза между покупками улучшений, с')
    ap.add_argument('--row-height', type=float, default=UPGRADE_ROW_HEIGHT,
                    help='высота одной строки магазина в экранных px (для прохода по улучшениям)')
    args = ap.parse_args()

    model_path = find_model_path(args.model)
    print(f'Модель: {model_path}')
    bot = CookieBot(model_path, args.conf, args.upgrade_cooldown, args.row_height)
    bot.run()


if __name__ == '__main__':
    main()
