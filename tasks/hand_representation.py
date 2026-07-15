# tasks/hand_representation.py
"""
Hand Representation Task
========================
Displays finger-position images, captures a webcam photo at the end of each
trial, and logs all results to a CSV file.

Output directory
----------------
    data/handrep/ID_{nom}/          ← CSV files
    data/handrep/ID_{nom}/photos/   ← trial + REF photos

Photo naming
------------
    Trial : S1_T6_ring_Z2.jpg
    Ref   : S1_REF1.jpg  /  S1_REF2.jpg

Sequence constraints (per finger)
---------------------------------
    3 imposed pairs Z1 → Z2  (consecutive)
    3 imposed pairs Z2 → Z1  (consecutive)
    4 isolated Z1 + 4 isolated Z2 (never adjacent to same finger other zone)

Finger / Zone mapping (source images = LEFT hand)
--------------------------------------------------
    Zone 1 : a2  a4  a6  a8  a10
    Zone 2 : a1  a3  a5  a7  a9

Image orientation
-----------------
    hand='gauche' : displayed as-is        → left hand
    hand='droite' : mirrored horizontally  → right hand
"""

import os
import cv2
import random
from datetime import datetime

from psychopy import visual, core, event

from utils.base_task import BaseTask
from tasks.sequence_generator import generate_sequence


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
IMAGES_DIR = os.path.join(_PROJECT_ROOT, "images")


class HandRepresentationTask(BaseTask):
    """PsychoPy task: hand-position display and webcam capture."""

    # Image mapping: (finger, zone) → image file
    IMAGE_MAP = {
        ("little", 2): "a1.png",
        ("little", 1): "a2.png",
        ("ring",   2): "a3.png",
        ("ring",   1): "a4.png",
        ("middle", 2): "a5.png",
        ("middle", 1): "a6.png",
        ("index",  2): "a7.png",
        ("index",  1): "a8.png",
        ("thumb",  2): "a9.png",
        ("thumb",  1): "a10.png",
    }

    BACKGROUND_COLOR = [0, 0, 0]

    FINGER_FLIP_MAP = {
        "thumb": "little",
        "index": "ring",
        "middle": "middle",
        "ring": "index",
        "little": "thumb",
    }

    BAR_Y = -0.75
    BAR_LEFT = -0.59
    BAR_MAX_WIDTH = 1.18
    BAR_TRACK_W = 1.2
    BAR_TRACK_H = 0.08
    BAR_FILL_H = 0.06

    BASE_RETURN_DURATION = 2.0

    CAPTURE_WIDTH = 1920
    CAPTURE_HEIGHT = 1080

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        win,
        nom,
        session="01",
        n_blocks=1,
        trial_duration=4.0,
        camera_index=1,
        hand="droite",
        enregistrer=True,
        images_dir=None,
        sequence_seed=None,
        **kwargs,
    ):
        folder_name = os.path.join("handrep", f"ID_{nom}")

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="HandRepresentation",
            folder_name=folder_name,
            eyetracker_actif=False,
            parport_actif=False,
            enregistrer=enregistrer,
            et_prefix="HND",
        )

        self.n_blocks = int(n_blocks)
        self.trial_duration = float(trial_duration)
        self.camera_index = int(camera_index)
        self.images_dir = images_dir or IMAGES_DIR
        self.sequence_seed = sequence_seed

        self.hand = hand.lower().strip()
        if self.hand not in ("droite", "gauche"):
            raise ValueError(f"'hand' must be 'droite' or 'gauche'. Got: '{hand}'")

        self.flip_horiz = self.hand == "droite"

        self.global_records = []
        self.camera = None
        self._global_trial_idx = 0

        self.photo_dir = os.path.join(self.data_dir, "photos")
        if self.enregistrer:
            os.makedirs(self.photo_dir, exist_ok=True)

        self.win.color = self.BACKGROUND_COLOR

        self._preload_images()
        self._setup_stimuli()
        self._init_incremental_file()
        self._log_startup()

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def _log_startup(self):
        if self.flip_horiz:
            hand_label = "DROITE (images miroir)"
        else:
            hand_label = "GAUCHE (images originales)"

        self.logger.ok("=" * 60)
        self.logger.ok("HAND REPRESENTATION TASK")
        self.logger.ok(f"Participant : {self.nom}  |  Session : {self.session}")
        self.logger.ok(f"Main        : {hand_label}")
        self.logger.ok(f"Blocs       : {self.n_blocks}  |  Durée essai : {self.trial_duration} s")
        self.logger.ok(f"Retour base : {self.BASE_RETURN_DURATION} s")
        self.logger.ok(f"Séquence    : contrainte (3×Z1Z2 + 3×Z2Z1 + 4 isolés par doigt)")
        self.logger.ok(f"Seed        : {self.sequence_seed}")
        self.logger.ok(f"Data dir    : {self.data_dir}")
        self.logger.ok(f"Photos dir  : {self.photo_dir}")
        self.logger.ok("=" * 60)

    def _preload_images(self):
        """Resolve and validate every image path."""
        self.loaded_images = {}
        for (finger, zone), img_file in self.IMAGE_MAP.items():
            img_path = os.path.join(self.images_dir, img_file)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            self.loaded_images[(finger, zone)] = img_path

    def _setup_stimuli(self):
        self.image_stim = visual.ImageStim(
            self.win,
            image=None,
            pos=(0, 0.1),
            size=(1.1, 1.1),
            flipHoriz=self.flip_horiz,
        )
        self.progress_track = visual.Rect(
            self.win,
            width=self.BAR_TRACK_W,
            height=self.BAR_TRACK_H,
            pos=(0, self.BAR_Y),
            lineColor=[0.6, 0.6, 0.6],
            lineWidth=2,
            fillColor=[0.3, 0.3, 0.3],
        )
        self.progress_fill = visual.Rect(
            self.win,
            width=0.001,
            height=self.BAR_FILL_H,
            pos=(self.BAR_LEFT, self.BAR_Y),
            lineColor=None,
            fillColor=[-1, -1, -1],
        )
        self.countdown_stim = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0),
            color=[1, 1, 1],
            height=0.3,
        )
        self.return_base_stim = visual.TextStim(
            self.win,
            text="Revenez à la position de base.",
            pos=(0, 0),
            color=[1, 1, 1],
            height=0.08,
            wrapWidth=1.8,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @property
    def _session_label(self):
        try:
            return str(int(self.session))
        except ValueError:
            return self.session

    def _get_displayed_finger(self, finger):
        if self.flip_horiz:
            return self.FINGER_FLIP_MAP.get(finger, finger)
        return finger

    # =========================================================================
    # CAMERA
    # =========================================================================

    def _open_camera(self):
        self.logger.log(f"Opening webcam (index={self.camera_index})")
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {self.camera_index}.")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAPTURE_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)

        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w != self.CAPTURE_WIDTH or actual_h != self.CAPTURE_HEIGHT:
            self.logger.warn(
                f"Requested {self.CAPTURE_WIDTH}×{self.CAPTURE_HEIGHT} "
                f"but got {actual_w}×{actual_h}."
            )
        else:
            self.logger.ok(f"Webcam: {actual_w}×{actual_h}")

        ret, frame = self.camera.read()
        if not ret or frame is None:
            self.camera.release()
            self.camera = None
            raise RuntimeError(
                f"Webcam opened but first read failed (index={self.camera_index})."
            )
        self.logger.ok(f"Webcam ready (index={self.camera_index})")

    def _close_camera(self):
        if self.camera is not None:
            try:
                self.camera.release()
                self.logger.log("Webcam closed.")
            except Exception:
                pass
            self.camera = None

    # =========================================================================
    # SEQUENCE GENERATION
    # =========================================================================

    def _build_block_trials(self, block_idx):
        """Generate 100 constrained trials for one block."""
        sequence = generate_sequence(seed=self.sequence_seed)

        trials = []
        for i, item in enumerate(sequence):
            finger = item["finger"]
            zone = item["zone"]
            image_file = self.IMAGE_MAP[(finger, zone)]

            trials.append({
                "block_idx": block_idx,
                "trial_in_block": i,
                "finger": finger,
                "zone": zone,
                "pair_type": item["pair_type"],
                "image_file": image_file,
                "position_label": f"{finger}_zone{zone}",
            })

        self.logger.ok(
            f"Block {block_idx + 1}: sequence generated "
            f"({len(trials)} trials, seed={self.sequence_seed})"
        )
        return trials

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def _show_instructions(self):
        hand_txt = "main gauche" if self.hand == "gauche" else "main droite"
        text = (
            f"Tâche de représentation de la main ({hand_txt})\n\n"
            "Une image indiquera un doigt et une zone.\n"
            "Placez votre doigt sur la zone indiquée\n"
            "avant la fin de la barre de progression.\n\n"
            "Maintenez la position jusqu'au message\n"
            "puis revenez à la position de repos.\n\n"
            "Appuyez sur une touche pour commencer."
        )
        self.show_instructions(text_override=text)

    def _show_ref_screen(self, message):
        try:
            self.instr_stim.text = message
            self.instr_stim.draw()
            self.win.flip()
        except Exception:
            msg = visual.TextStim(
                self.win,
                text=message,
                pos=(0, 0),
                color=[1, 1, 1],
                height=0.05,
                wrapWidth=1.8,
            )
            msg.draw()
            self.win.flip()

    def _wait_for_space(self, message=None):
        if message is not None:
            self._show_ref_screen(message)
        event.clearEvents()
        event.waitKeys(keyList=["space"])

    def _show_countdown(self, seconds=3):
        for count in range(seconds, 0, -1):
            self.countdown_stim.text = str(count)
            self.countdown_stim.draw()
            self.win.flip()
            core.wait(1.0)

    def _draw_progress_screen(self, image_path, elapsed, duration):
        progress = min(max(elapsed / duration, 0.0), 1.0)
        fill_w = max(0.001, self.BAR_MAX_WIDTH * progress)

        self.image_stim.image = image_path
        self.progress_fill.width = fill_w
        self.progress_fill.pos = (self.BAR_LEFT + fill_w * 0.5, self.BAR_Y)

        self.image_stim.draw()
        self.progress_track.draw()
        self.progress_fill.draw()
        self.win.flip()

    def _show_return_to_base(self):
        self.return_base_stim.draw()
        self.win.flip()
        core.wait(self.BASE_RETURN_DURATION)

    # =========================================================================
    # PHOTO CAPTURE
    # =========================================================================

    def _build_photo_filename(self, trial):
        finger = self._get_displayed_finger(trial["finger"])
        return (
            f"S{self._session_label}"
            f"_T{self._global_trial_idx + 1}"
            f"_{finger}"
            f"_Z{trial['zone']}.jpg"
        )

    def _build_ref_photo_filename(self, ref_number):
        return f"S{self._session_label}_REF{ref_number}.jpg"

    def _read_camera_frame(self):
        frame = None
        for _ in range(3):
            ret, current = self.camera.read()
            if ret and current is not None:
                frame = current
        return frame

    def _capture_photo(self, trial):
        if self.camera is None:
            raise RuntimeError("Webcam is not initialised.")
        frame = self._read_camera_frame()
        if frame is None:
            raise RuntimeError("Webcam capture failed after 3 attempts.")
        filename = self._build_photo_filename(trial)
        save_path = os.path.join(self.photo_dir, filename)
        if not cv2.imwrite(save_path, frame):
            raise RuntimeError(f"Could not save photo: {save_path}")
        return save_path, filename

    def _capture_ref_photo(self, filename):
        if self.camera is None:
            raise RuntimeError("Webcam is not initialised.")
        frame = self._read_camera_frame()
        if frame is None:
            raise RuntimeError("Reference capture failed after 3 attempts.")
        save_path = os.path.join(self.photo_dir, filename)
        if not cv2.imwrite(save_path, frame):
            raise RuntimeError(f"Could not save reference photo: {save_path}")
        self.logger.ok(f"Reference photo: {filename}")
        return save_path

    def _capture_ref_with_countdown(self, filename, post_capture_message):
        self._wait_for_space(
            "Photo de référence.\n\n"
            "Placez votre main au repos\n"
            "puis appuyez sur ESPACE."
        )
        self._show_countdown(seconds=3)
        self._show_ref_screen("Capture...")
        core.wait(0.1)
        self._capture_ref_photo(filename)
        print(f"  [REF] {filename}")
        self._wait_for_space(post_capture_message)

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log_trial(self, trial, image_path, photo_path, photo_filename,
                   image_onset, capture_time):
        record = {
            "participant": self.nom,
            "session": self.session,
            "task_name": self.task_name,
            "hand": self.hand,
            "flip_horiz": self.flip_horiz,
            "block_idx": trial["block_idx"],
            "block_number": trial["block_idx"] + 1,
            "trial_in_block": trial["trial_in_block"],
            "global_trial": self._global_trial_idx + 1,
            "finger_source": trial["finger"],
            "finger_displayed": self._get_displayed_finger(trial["finger"]),
            "zone": trial["zone"],
            "pair_type": trial["pair_type"],
            "image_file": trial["image_file"],
            "image_path": image_path,
            "photo_filename": photo_filename,
            "photo_path": photo_path,
            "image_onset": round(image_onset, 4),
            "capture_time_task": round(capture_time, 4),
            "trial_duration": self.trial_duration,
            "wall_timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"),
        }
        self.global_records.append(record)
        self.save_trial_incremental(record)

    def _print_trial_summary(self, trial, capture_time, photo_filename):
        displayed_finger = self._get_displayed_finger(trial["finger"])
        pair_tag = trial["pair_type"][:3].upper()
        print(
            f"  B{trial['block_idx'] + 1:02d} "
            f"T{self._global_trial_idx + 1:03d} | "
            f"{self.hand[0].upper()} | "
            f"{displayed_finger:<8} Z{trial['zone']} | "
            f"{pair_tag} | "
            f"t={capture_time:7.3f} s | "
            f"{photo_filename}"
        )

    # =========================================================================
    # TRIAL & BLOCK
    # =========================================================================

    def run_trial(self, trial):
        image_path = self.loaded_images[(trial["finger"], trial["zone"])]
        onset = self.task_clock.getTime()
        trial_clock = core.Clock()

        while trial_clock.getTime() < self.trial_duration:
            self._draw_progress_screen(
                image_path=image_path,
                elapsed=trial_clock.getTime(),
                duration=self.trial_duration,
            )
            self.get_keys(key_list=[])

        capture_time = self.task_clock.getTime()
        photo_path, photo_fn = self._capture_photo(trial)

        self._log_trial(
            trial=trial,
            image_path=image_path,
            photo_path=photo_path,
            photo_filename=photo_fn,
            image_onset=onset,
            capture_time=capture_time,
        )
        self._print_trial_summary(trial, capture_time, photo_fn)
        self._show_return_to_base()
        self._global_trial_idx += 1

    def run_block(self, block_idx):
        trials = self._build_block_trials(block_idx)
        print(f"\n{'=' * 60}")
        print(f"BLOCK {block_idx + 1}/{self.n_blocks} — {len(trials)} trials — {self.hand}")
        print("=" * 60)
        self.logger.log(f"START BLOCK {block_idx + 1}")

        ref1_fn = self._build_ref_photo_filename(1)
        self._capture_ref_with_countdown(
            filename=ref1_fn,
            post_capture_message=(
                "Référence initiale enregistrée.\n\n"
                "Appuyez sur ESPACE pour commencer."
            ),
        )

        for trial in trials:
            self.run_trial(trial)

        ref2_fn = self._build_ref_photo_filename(2)
        self._capture_ref_with_countdown(
            filename=ref2_fn,
            post_capture_message=(
                "Référence finale enregistrée.\n\n"
                "Appuyez sur ESPACE pour continuer."
            ),
        )

        self.logger.log(f"END BLOCK {block_idx + 1}")

    # =========================================================================
    # SESSION
    # =========================================================================

    def _start_session(self):
        self._show_instructions()
        self.task_clock.reset()
        self.logger.ok("Session started.")

    def _end_session(self):
        saved_path = self.save_data(
            data_list=self.global_records,
            filename_suffix="_final",
        )
        try:
            if self.win and not self.win._closed:
                self.instr_stim.text = "Fin de la session. Merci."
                self.instr_stim.draw()
                self.win.flip()
                core.wait(2.0)
        except Exception:
            pass
        return saved_path

    # =========================================================================
    # ENTRY POINT
    # =========================================================================

    def run(self):
        saved_path = None
        try:
            self._open_camera()
            self._start_session()
            for block_idx in range(self.n_blocks):
                self.run_block(block_idx)
            self.logger.ok("Task completed.")
        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Manual interruption.")
        except Exception as e:
            self.logger.err(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._close_camera()
            saved_path = self._end_session()
        return saved_path