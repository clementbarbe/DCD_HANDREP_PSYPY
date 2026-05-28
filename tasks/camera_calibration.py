# tasks/camera_calibration.py
"""
Camera Calibration Task
=======================
Lance _calibration_ui.py en subprocess indépendant pour chaque surface.
Aucune fenêtre PsychoPy n'est créée.

Sortie : S{session}_{table|plateau}_calibration.png  (1920×1080)
         dans data/CameraCalibration/
"""

import os
import sys
import shutil
import subprocess
import tempfile

from utils.logger import get_logger

_UI_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_calibration_ui.py")


def _find_max_camera_index(limit=10):
    """Teste les index 0..limit et retourne le plus élevé fonctionnel."""
    import cv2
    best = -1
    for idx in range(limit + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                best = idx
            cap.release()
    return best


class CameraCalibrationTask:
    """Lance la calibration dans un subprocess PyQt6 dédié.
    
    Sortie : un PNG 1920×1080 par surface (pas de JSON).
    La caméra avec l'index le plus élevé est automatiquement sélectionnée.
    """

    def __init__(
        self,
        win,
        nom,
        session="01",
        camera_index=-1,
        enregistrer=True,
        **kwargs,
    ):
        self.nom = str(nom)
        self.session = str(session)
        self.enregistrer = enregistrer
        self.results = {}

        self.logger = get_logger()

        # Auto-detect max camera index
        if camera_index < 0:
            self.camera_index = _find_max_camera_index()
            if self.camera_index < 0:
                raise RuntimeError("No camera found.")
            self.logger.ok(f"Auto-selected camera index: {self.camera_index}")
        else:
            self.camera_index = int(camera_index)

        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(_root, "data", "CameraCalibration")
        if self.enregistrer:
            os.makedirs(self.data_dir, exist_ok=True)

        self.logger.ok("=" * 60)
        self.logger.ok("CAMERA CALIBRATION — READY")
        self.logger.ok(f"Participant : {self.nom}  |  Session : {self.session}")
        self.logger.ok(f"Camera index: {self.camera_index}")
        self.logger.ok("=" * 60)

    @property
    def _session_label(self):
        try:
            return str(int(self.session))
        except ValueError:
            return self.session

    def run(self, calibration_types=("table", "plateau"), flip_feed=False):
        """Lance un subprocess par surface. Retourne {type: png_path|None}."""
        try:
            for cal_type in calibration_types:
                self.logger.log(f"Starting calibration: {cal_type.upper()}")

                # Fichier temporaire PNG
                tmp_png = os.path.join(
                    tempfile.gettempdir(),
                    f"calibration_{cal_type}_{os.getpid()}.png",
                )
                if os.path.exists(tmp_png):
                    os.unlink(tmp_png)

                cmd = [
                    sys.executable, _UI_SCRIPT,
                    "--type", cal_type,
                    "--camera", str(self.camera_index),
                    "--output", tmp_png,
                ]
                if flip_feed:
                    cmd.append("--flip")

                try:
                    proc = subprocess.run(cmd, timeout=600)

                    if proc.returncode == 0 and os.path.exists(tmp_png):
                        self.logger.ok(f"Calibration '{cal_type}' confirmed.")

                        if self.enregistrer:
                            fname = f"S{self._session_label}_{cal_type}_calibration.png"
                            dest = os.path.join(self.data_dir, fname)
                            os.makedirs(self.data_dir, exist_ok=True)
                            shutil.copy2(tmp_png, dest)
                            self.results[cal_type] = dest
                            self.logger.ok(f"Saved → {fname}")
                        else:
                            self.results[cal_type] = tmp_png
                    else:
                        self.logger.warn(
                            f"Calibration '{cal_type}' cancelled "
                            f"(returncode={proc.returncode})."
                        )
                        self.results[cal_type] = None

                except subprocess.TimeoutExpired:
                    self.logger.warn(f"Calibration '{cal_type}' timed out.")
                    self.results[cal_type] = None

                finally:
                    if os.path.exists(tmp_png):
                        os.unlink(tmp_png)

            self.logger.ok("Calibration session completed.")

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Manual interruption.")
        except Exception as e:
            self.logger.err(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

        return self.results