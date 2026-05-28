# tasks/_calibration_ui.py
"""
Script standalone lancé en subprocess par CameraCalibrationTask.
Crée sa propre QApplication — aucun conflit avec le process principal.

Usage interne :
    python _calibration_ui.py --type table --camera 0
                               --nom X --session 1 --output /tmp/calib.png [--flip]

Sortie : une capture PNG 1920×1080 (sans overlay).
"""
import sys
import os
import argparse

import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


# ── Résolution imposée ───────────────────────────────────────────────────────
CAMERA_W = 1920
CAMERA_H = 1080
CAMERA_FPS = 10

CALIBRATION_LABELS = {
    "table": "Table",
    "plateau": "Plateau",
}

RATIO = 0.8
DISPLAY_W = round(CAMERA_W * RATIO)
DISPLAY_H = round(CAMERA_H * RATIO)


def _find_max_camera_index(limit=10):
    """Teste les index 0..limit et retourne le plus élevé qui s'ouvre."""
    best = -1
    for idx in range(limit + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                best = idx
            cap.release()
    return best


def _open_camera(index: int) -> cv2.VideoCapture:
    """Ouvre la caméra et force 1920×1080."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return cap

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (CAMERA_W, CAMERA_H):
        print(
            f"[CALIB] Résolution demandée {CAMERA_W}×{CAMERA_H}, "
            f"obtenue {actual_w}×{actual_h}",
            file=sys.stderr,
        )
    return cap


class CalibrationWindow(QMainWindow):

    def __init__(self, camera, cal_type, flip_feed, output_path):
        super().__init__()
        self.camera = camera
        self.cal_type = cal_type
        self.label = CALIBRATION_LABELS[cal_type]
        self.flip_feed = flip_feed
        self.output_path = output_path
        self.confirmed = False

        self.center_x = DISPLAY_W // 2
        self.center_y = DISPLAY_H // 2

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(33)

    def _build_ui(self):
        self.setWindowTitle(f"Calibration — {self.label}")
        self.setStyleSheet("background:#1a1a1a; color:white;")

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setSpacing(8)
        lay.setContentsMargins(12, 12, 12, 12)

        title = QLabel(f"Calibration — {self.label}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:15px; color:#FFE066; padding:6px;")
        lay.addWidget(title)

        res_label = QLabel(
            f"Caméra : {CAMERA_W}×{CAMERA_H}  |  "
            f"Affichage : {DISPLAY_W}×{DISPLAY_H}"
        )
        res_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_label.setStyleSheet("font-size:11px; color:#88aaff; padding:2px;")
        lay.addWidget(res_label)

        instr = QLabel("Centrez le point rouge sur votre repère, puis confirmez.")
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instr.setStyleSheet("font-size:12px; color:#cccccc; padding:4px;")
        lay.addWidget(instr)

        self.feed_label = QLabel()
        self.feed_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        self.feed_label.setStyleSheet("background:black; border:1px solid #444;")
        lay.addWidget(self.feed_label, alignment=Qt.AlignmentFlag.AlignCenter)

        row = QHBoxLayout()
        btn_ok = QPushButton("Confirmer  [Entrée]")
        btn_ok.setStyleSheet(
            "background:#2a7a2a; color:white; padding:8px 28px; font-size:13px;"
        )
        btn_ok.clicked.connect(self._confirm)

        btn_cancel = QPushButton("Annuler  [Échap]")
        btn_cancel.setStyleSheet(
            "background:#7a2a2a; color:white; padding:8px 28px; font-size:13px;"
        )
        btn_cancel.clicked.connect(self.close)

        row.addStretch()
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        row.addStretch()
        lay.addLayout(row)

        self.adjustSize()

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._confirm()
        elif k == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def _update_frame(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return
        if self.flip_feed:
            frame = cv2.flip(frame, 0)

        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        # Point rouge uniquement sur le flux live (pas sur le PNG sauvé)
        cv2.circle(display, (self.center_x, self.center_y), 6, (0, 0, 255), -1, cv2.LINE_AA)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.feed_label.setPixmap(QPixmap.fromImage(qimg))

    def _confirm(self):
        self._timer.stop()

        # Capturer une frame native propre (sans overlay)
        native_frame = None
        for _ in range(3):
            ret, frame = self.camera.read()
            if ret and frame is not None:
                native_frame = frame

        if self.flip_feed and native_frame is not None:
            native_frame = cv2.flip(native_frame, 0)

        if native_frame is not None:
            h, w = native_frame.shape[:2]
            if (w, h) != (CAMERA_W, CAMERA_H):
                native_frame = cv2.resize(native_frame, (CAMERA_W, CAMERA_H))
            cv2.imwrite(self.output_path, native_frame)
            self.confirmed = True
        else:
            print("[CALIB] Capture failed at confirmation.", file=sys.stderr)

        self.close()

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["table", "plateau"])
    ap.add_argument("--camera", type=int, default=-1)
    ap.add_argument("--output", required=True)
    ap.add_argument("--flip", action="store_true")
    args = ap.parse_args()

    # Force max camera index
    if args.camera < 0:
        args.camera = _find_max_camera_index()
        if args.camera < 0:
            print("[CALIB] No camera found.", file=sys.stderr)
            sys.exit(1)
        print(f"[CALIB] Auto-selected camera index: {args.camera}")

    cap = _open_camera(args.camera)
    if not cap.isOpened():
        print(f"[CALIB] Cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    ret, _ = cap.read()
    if not ret:
        cap.release()
        print("[CALIB] Camera read failed", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    win = CalibrationWindow(
        camera=cap,
        cal_type=args.type,
        flip_feed=args.flip,
        output_path=args.output,
    )
    win.show()
    win.raise_()
    win.activateWindow()
    app.exec()

    cap.release()
    sys.exit(0 if win.confirmed else 1)