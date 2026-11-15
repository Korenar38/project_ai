"""Помощник учителя: подсказывает, на каких учеников стоит обратить внимание.

Использует нейросеть YOLO (модель позы) и камеру. Программа:
  * НЕ распознаёт лица и не определяет, кто именно ученик (только номер на экране);
  * НЕ сохраняет видео и кадры (в журнал пишутся только время, номер и тип сигнала);
  * НЕ ставит диагнозов: сигнал означает лишь «стоит присмотреться».
Запуск:  python detector.py --source 0      (0 - первая камера, либо путь к видеофайлу)
Выход:   клавиша q.
"""
import argparse
import csv
import math
import time
from collections import defaultdict, deque

# Названия сигналов: для журнала (русский) и для экрана (латиница: OpenCV не рисует кириллицу).
SIGNALS = {
    "move": ("много двигается или ходит по классу", "moves a lot"),
    "head": ("долго не смотрит вперёд (голова опущена или отвёрнута)", "head down/away"),
    "alone": ("долго держится в стороне от других", "keeps apart"),
    "sharp": ("резкие движения рядом с другим учеником", "sharp moves near others"),
}
NOSE, L_SH, R_SH, L_WR, R_WR = 0, 5, 6, 9, 10  # номера точек тела в модели YOLO-pose (COCO)


def head_down(k, conf=0.3, ratio=0.25):
    """True, если голова опущена или отвёрнута (нос не виден или почти на уровне плеч)."""
    ls, rs, nose = k[L_SH], k[R_SH], k[NOSE]
    if ls[2] < conf or rs[2] < conf:
        return False  # плечи не видны, судить нельзя
    width = abs(ls[0] - rs[0])
    if width < 1:
        return False
    if nose[2] < conf:
        return True
    return ((ls[1] + rs[1]) / 2 - nose[1]) / width < ratio


def wrists(k, conf=0.3):
    return [(k[i][0], k[i][1]) if k[i][2] >= conf else None for i in (L_WR, R_WR)]


class ClassroomWatcher:
    """Копит историю по каждому ученику и выдаёт сигналы-подсказки."""

    def __init__(self, window=30.0, cooldown=60.0, move_thr=0.25, head_share=0.7,
                 alone_dist=2.5, alone_share=0.8, sharp_speed=1.5, sharp_near=1.0, sharp_count=3):
        self.window, self.cooldown = window, cooldown
        self.move_thr, self.head_share = move_thr, head_share
        self.alone_dist, self.alone_share = alone_dist, alone_share
        self.sharp_speed, self.sharp_near, self.sharp_count = sharp_speed, sharp_near, sharp_count
        self.hist = defaultdict(deque)
        self.last_sent = {}

    def update(self, t, people):
        """people: {номер: (рамка (x1,y1,x2,y2), точки [(x,y,уверенность)*17])}.
        Возвращает список (номер, ключ_сигнала)."""
        centers = {i: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for i, (b, _) in people.items()}
        out = []
        for i, (box, k) in people.items():
            h = max(box[3] - box[1], 1.0)
            c = centers[i]
            others = [math.dist(c, centers[j]) / h for j in people if j != i]
            near = min(others) if others else math.inf
            w = wrists(k)
            hist = self.hist[i]
            speed = 0.0
            if hist:
                p = hist[-1]
                dt = max(t - p["t"], 1e-3)
                for a, b in zip(w, p["w"]):
                    if a and b:
                        speed = max(speed, math.dist(a, b) / dt / h)
            hist.append({"t": t, "c": c, "h": h, "head": head_down(k), "near": near,
                         "w": w, "sharp": speed > self.sharp_speed and near < self.sharp_near})
            while hist and t - hist[0]["t"] > self.window:
                hist.popleft()
            span = t - hist[0]["t"]
            if span < self.window * 0.5 or len(hist) < 5:
                continue
            path = sum(math.dist(a["c"], b["c"]) for a, b in zip(hist, list(hist)[1:]))
            avg_h = sum(e["h"] for e in hist) / len(hist)
            found = {
                "move": path / span / avg_h > self.move_thr,
                "head": sum(e["head"] for e in hist) / len(hist) > self.head_share,
                "alone": sum(e["near"] > self.alone_dist for e in hist) / len(hist) > self.alone_share
                         and near != math.inf,
                "sharp": sum(e["sharp"] for e in hist) >= self.sharp_count,
            }
            for key, on in found.items():
                if on and t - self.last_sent.get((i, key), -1e9) >= self.cooldown:
                    self.last_sent[(i, key)] = t
                    out.append((i, key))
        for gone in [i for i in self.hist if i not in people and self.hist[i]
                     and t - self.hist[i][-1]["t"] > self.window]:
            del self.hist[gone]
        return out


def main():
    ap = argparse.ArgumentParser(description="Помощник учителя на основе YOLO")
    ap.add_argument("--source", default="0", help="номер камеры или путь к видеофайлу")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="модель YOLO для позы")
    ap.add_argument("--window", type=float, default=30.0, help="окно наблюдения, секунд")
    ap.add_argument("--log", default="signals_log.csv", help="файл журнала сигналов")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    watcher = ClassroomWatcher(window=args.window)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    alerts = []  # (до какого времени показывать, текст)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            res = model.track(frame, persist=True, conf=0.4, verbose=False)[0]
            people = {}
            if res.boxes.id is not None and res.keypoints is not None:
                for i, b, k in zip(res.boxes.id.int().tolist(), res.boxes.xyxy.tolist(),
                                   res.keypoints.data.tolist()):
                    people[i] = (b, k)
            now = time.time()
            for pid, key in watcher.update(now, people):
                print(f"{time.strftime('%H:%M:%S')}  ученик №{pid}: {SIGNALS[key][0]} — стоит присмотреться")
                writer.writerow([time.strftime("%H:%M:%S"), pid, key])
                f.flush()
                alerts.append((now + 8, f"ID {pid}: {SIGNALS[key][1]} - take a look"))
            for pid, (b, _) in people.items():
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 163, 15), 2)
                cv2.putText(frame, f"ID {pid}", (int(b[0]), max(int(b[1]) - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 163, 15), 2)
            alerts = [a for a in alerts if a[0] > now]
            for n, (_, text) in enumerate(alerts[-4:]):
                cv2.putText(frame, text, (10, 25 + 25 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Teacher assistant (q - quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
