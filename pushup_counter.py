#!/usr/bin/env python3
"""Webcam push-up counter using YOLO pose estimation.

A side view works best: put the camera low, beside you, so one full arm
is in frame. The badge in the top-left counts each time you lower down
and press back up.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python pushup_counter.py

Keys: q quit, r reset the count. The first run downloads the pose weights.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# COCO body keypoints used by YOLO pose models.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
ARMS = (
    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
)

# Elbow angle is ~180 with a straight arm and ~70-90 at the bottom of a push-up.
DOWN_ANGLE = 110.0
UP_ANGLE = 150.0


def joint_angle(a, b, c) -> float | None:
    """Interior angle at point b, in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def elbow_angle_from_pose(xy, conf, min_conf: float = 0.45) -> float | None:
    """Confidence-weighted elbow angle for the visible arm or arms."""
    if xy is None or len(xy) <= RIGHT_WRIST:
        return None

    scored: list[tuple[float, float]] = []
    for shoulder, elbow, wrist in ARMS:
        score = float(min(conf[shoulder], conf[elbow], conf[wrist]))
        if score < min_conf:
            continue
        angle = joint_angle(xy[shoulder], xy[elbow], xy[wrist])
        if angle is not None:
            scored.append((score, angle))

    if not scored:
        return None
    weight = sum(score for score, _ in scored)
    return sum(score * angle for score, angle in scored) / weight


class PushUpCounter:
    """Counts a rep on the way back up, after the elbows have been bent."""

    def __init__(
        self,
        down_angle: float = DOWN_ANGLE,
        up_angle: float = UP_ANGLE,
        smoothing: float = 0.45,
    ):
        self.down_angle = down_angle
        self.up_angle = up_angle
        self.smoothing = smoothing
        self.count = 0
        self.stage = "up"
        self.angle: float | None = None
        self._missed = 0

    def reset(self) -> None:
        self.count = 0
        self.stage = "up"
        self.angle = None
        self._missed = 0

    def update(self, angle: float | None) -> int:
        if angle is None:
            self._missed += 1
            # Drop a half-finished rep if the body leaves the frame.
            if self._missed >= 15:
                self.stage = "up"
                self.angle = None
            return self.count

        self._missed = 0
        if self.angle is None:
            self.angle = float(angle)
        else:
            self.angle = self.smoothing * float(angle) + (1.0 - self.smoothing) * self.angle

        if self.stage == "up" and self.angle <= self.down_angle:
            self.stage = "down"
        elif self.stage == "down" and self.angle >= self.up_angle:
            self.stage = "up"
            self.count += 1
        return self.count


def largest_person(result):
    """Keypoints for the largest detected person, or None."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or keypoints.xy.shape[0] == 0:
        return None

    xy_all = keypoints.xy
    index = 0
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) == xy_all.shape[0]:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        index = int(areas.argmax())

    xy = xy_all[index].detach().cpu().numpy()
    if keypoints.conf is not None:
        conf = keypoints.conf[index].detach().cpu().numpy()
    else:
        conf = np.ones(xy.shape[0], dtype=float)
    return xy, conf


def draw_counter(frame, count: int, tracking: bool) -> None:
    """Small count badge in the top-left corner."""
    number = str(count)
    label = "push-ups" if tracking else "looking..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    number_scale, number_thickness = 0.85, 2
    label_scale, label_thickness = 0.42, 1

    (number_w, number_h), number_base = cv2.getTextSize(number, font, number_scale, number_thickness)
    (label_w, label_h), _ = cv2.getTextSize(label, font, label_scale, label_thickness)

    pad_x, pad_y = 12, 8
    width = max(number_w, label_w) + pad_x * 2
    height = number_h + label_h + number_base + pad_y * 2 + 4
    x, y = 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 210, 130), 1)

    number_color = (255, 255, 255) if tracking else (180, 180, 180)
    label_color = (120, 220, 150) if tracking else (160, 160, 160)
    cv2.putText(
        frame,
        number,
        (x + pad_x, y + pad_y + number_h),
        font,
        number_scale,
        number_color,
        number_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x + pad_x, y + height - pad_y),
        font,
        label_scale,
        label_color,
        label_thickness,
        cv2.LINE_AA,
    )


def draw_hint(frame) -> None:
    hint = "side view   ·   r reset   ·   q quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (12, frame.shape[0] - 16)
    cv2.putText(frame, hint, (origin[0] + 1, origin[1] + 1), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hint, origin, font, 0.45, (235, 235, 235), 1, cv2.LINE_AA)


def open_camera(index: int) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(index)


def run(camera_index: int, model_name: str) -> None:
    print(f"Loading {model_name} (first run downloads the weights)...")
    model = YOLO(model_name)

    capture = open_camera(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        print(
            "Could not open the camera. On macOS, open System Settings → "
            "Privacy & Security → Camera and allow Cursor or Terminal, then run this again.",
            file=sys.stderr,
        )
        sys.exit(1)

    counter = PushUpCounter()
    window = "Push-up Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Camera open. Do push-ups from the side. Press q to quit, r to reset.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("The camera stopped sending frames.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            results = model.predict(frame, verbose=False, imgsz=640, conf=0.35, max_det=2)
            result = results[0] if results else None

            view = frame
            angle = None
            if result is not None:
                person = largest_person(result)
                if person is not None:
                    xy, conf = person
                    angle = elbow_angle_from_pose(xy, conf)
                    view = result.plot(boxes=False, labels=False, conf=False)

            previous = counter.count
            counter.update(angle)
            if counter.count != previous:
                print(f"Push-ups: {counter.count}", flush=True)

            draw_counter(view, counter.count, tracking=angle is not None)
            draw_hint(view)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                counter.reset()
                print("Count reset.", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_self_test() -> None:
    straight = joint_angle((100, 100), (100, 200), (100, 300))
    bent = joint_angle((100, 100), (100, 200), (200, 200))
    assert straight is not None and abs(straight - 180) < 1
    assert bent is not None and abs(bent - 90) < 1

    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)
    xy[LEFT_SHOULDER], xy[LEFT_ELBOW], xy[LEFT_WRIST] = (100, 100), (100, 200), (100, 300)
    conf[LEFT_SHOULDER] = conf[LEFT_ELBOW] = conf[LEFT_WRIST] = 0.9
    assert abs(elbow_angle_from_pose(xy, conf) - 180) < 1

    counter = PushUpCounter(smoothing=1.0)
    assert counter.update(None) == 0
    for angle in (170, 140, 100, 80, 100, 140, 170):
        counter.update(angle)
    assert counter.count == 1
    assert counter.stage == "up"

    for _ in range(2):
        counter.update(80)
        counter.update(170)
    assert counter.count == 3

    # A shallow dip between the two thresholds is not a rep.
    counter.update(130)
    counter.update(170)
    assert counter.count == 3

    counter.reset()
    assert counter.count == 0
    print("Self-test passed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Count push-ups from your webcam with YOLO pose.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="YOLO pose weights")
    parser.add_argument("--self-test", action="store_true", help="Check the rep counter and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    run(args.camera, args.model)


if __name__ == "__main__":
    main()
