# motor_planning.py
"""
Motor Planning Protocol — Grasp / Touch  ×  Hand / Tool
========================================================
HD-EEG — UN run par lancement.

ARCHITECTURE : PRE-COMPUTED TIMELINE
─────────────────────────────────────
1. À l'initialisation, TOUS les événements (visuels, sonores, triggers,
   marqueurs) sont pré-calculés dans une timeline unique, triée par onset.
2. Après le trigger de démarrage, le moteur d'exécution parcourt la
   timeline : attente haute-précision pour les sons, standard pour le
   reste.
3. Zéro calcul de séquence ou de jitter pendant l'acquisition.

Structure d'un essai :
    Preview  →  Cue (« Grasp » / « Touch »)  →  Plan (5.5 s ± 0.5 s)
             →  Go beep  →  Execute (2 s)  →  ITI (8 s)

Design :
    8 runs au total, 4 par effecteur (main / outil).
    Par run : 20 essais grasp + 20 essais touch, ordre pseudo-aléatoire
    (pas de répétition immédiate de condition).

Fichiers produits :
    *_planned.csv       → timeline pré-calculée (avant exécution)
    *_incremental.csv   → écriture événement-par-événement pendant le run
    *_<timestamp>.csv   → fichier final propre (planned + actual)
"""

from __future__ import annotations

import csv
import gc
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from psychopy import core, visual
from psychopy import sound as psysound
from utils.base_task import BaseTask

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

CONDITIONS: List[str] = ["grasp", "touch"]
EFFECTORS:  List[str] = ["hand", "tool"]

# Codes port parallèle pour marqueurs EEG (personnalisables via __init__)
DEFAULT_EVENT_CODES: Dict[str, int] = {
    "run_start":      100,
    "run_end":        200,
    "trial_start":      1,
    "preview_start":    2,
    "cue_grasp":       10,
    "cue_touch":       20,
    "go_grasp":        30,
    "go_touch":        40,
    "execute_start":   50,
    "iti_start":       60,
}

# Priorité de tri quand plusieurs événements partagent le même onset
_ACTION_PRIORITY: Dict[str, int] = {
    "visual_fixation":     0,   # flip d'abord
    "visual_instruction":  0,
    "parport_event":       1,   # puis triggers EEG
    "sound_cue":           2,   # puis sons (haute précision)
    "sound_go":            2,
    "marker":              3,   # puis marqueurs logiques
}


# ═════════════════════════════════════════════════════════════════════════════

class MotorPlanning(BaseTask):
    """
    One run per instantiation.

    Parameters
    ----------
    effector : str
        'hand' ou 'tool' — détermine les instructions et est enregistré
        dans les données. Constant pour tout le run.
    run_number : int
        Numéro du run (1-8), assigné depuis le menu.
    plan_duration : float
        Durée totale de la phase Plan **depuis l'onset du cue** jusqu'à
        l'onset du go beep (en secondes). Jittée ± plan_jitter.
    """

    def __init__(
        self,
        win: visual.Window,
        nom: str,
        session: str = "01",
        mode: str = "eeg",
        effector: str = "hand",
        run_number: int = 1,
        # ── nombre d'essais ──
        n_trials_per_condition: int = 20,
        # ── timing (en secondes) ──
        preview_duration: float = 2.0,
        cue_duration: float = 0.5,
        plan_duration: float = 5.5,
        plan_jitter: float = 0.5,
        go_duration: float = 0.5,
        execute_duration: float = 2.0,
        iti_duration: float = 8.0,
        # ── baselines ──
        initial_baseline: float = 10.0,
        final_baseline: float = 10.0,
        # ── sons ──
        sound_dir: str = "sounds",
        grasp_sound_file: str = "grasp.wav",
        touch_sound_file: str = "touch.wav",
        go_beep_freq: float = 1000.0,
        # ── misc ──
        enregistrer: bool = True,
        eyetracker_actif: bool = False,
        parport_actif: bool = True,
        event_codes: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Motor_Planning",
            folder_name="motor_planning",
            eyetracker_actif=eyetracker_actif,
            parport_actif=parport_actif,
            enregistrer=enregistrer,
            et_prefix="MP",
        )

        # ── identifiers ──────────────────────────────────────────────────
        self.mode: str       = mode.lower()
        self.effector: str   = effector.lower()
        self.run_number: int = run_number

        # ── trial design ─────────────────────────────────────────────────
        self.n_trials_per_condition: int = n_trials_per_condition
        self.n_trials_total: int = len(CONDITIONS) * n_trials_per_condition

        # ── timing ───────────────────────────────────────────────────────
        self.preview_duration: float = max(0.5, preview_duration)
        self.cue_duration: float     = cue_duration
        self.plan_duration: float    = plan_duration
        self.plan_jitter: float      = plan_jitter
        self.go_duration: float      = go_duration
        self.execute_duration: float = execute_duration
        self.iti_duration: float     = iti_duration
        self.initial_baseline: float = initial_baseline
        self.final_baseline: float   = final_baseline

        # ── sons ─────────────────────────────────────────────────────────
        self.sound_dir: str          = sound_dir
        self.grasp_sound_file: str   = grasp_sound_file
        self.touch_sound_file: str   = touch_sound_file
        self.go_beep_freq: float     = go_beep_freq

        # ── event codes EEG ──────────────────────────────────────────────
        self.event_codes: Dict[str, int] = (
            event_codes if event_codes else dict(DEFAULT_EVENT_CODES)
        )

        # ── runtime state ────────────────────────────────────────────────
        self.global_records: List[Dict[str, Any]] = []

        # ═══ PRE-COMPUTED TIMELINE ═══
        self.timeline: List[Dict[str, Any]] = []

        # ── init chain ───────────────────────────────────────────────────
        self._detect_display_scaling()
        self._measure_frame_rate()
        self._setup_key_mapping()
        self._setup_visual_stimuli()
        self._setup_sounds()
        self._init_incremental_file(
            suffix=f"_{self.effector}_run{self.run_number:02d}"
        )

        # ── BUILD THE ENTIRE TIMELINE ────────────────────────────────────
        self._build_full_timeline()
        self._save_planned_timeline()

        self.logger.ok(
            f"MotorPlanning ready | {self.effector} "
            f"run {self.run_number:02d} | "
            f"{self.n_trials_total} trials | "
            f"{self.frame_rate:.1f} Hz | "
            f"frame = {self.frame_duration_s * 1000:.1f} ms | "
            f"{len(self.timeline)} events pre-computed | "
            f"~{self.timeline[-1]['onset_s']:.1f} s "
            f"({self.timeline[-1]['onset_s'] / 60:.1f} min)"
        )

    # =====================================================================
    #  INIT HELPERS
    # =====================================================================

    def _detect_display_scaling(self) -> None:
        self.pixel_scale = 2.0 if self.win.size[1] > 1200 else 1.0

    def _measure_frame_rate(self) -> None:
        measured = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1
        )
        self.frame_rate = measured if measured else 60.0
        self.frame_duration_s  = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 / self.frame_rate
        self.logger.log(
            f"Frame rate: {self.frame_rate:.1f} Hz → "
            f"{self.frame_duration_s * 1000:.2f} ms/frame"
        )

    def _setup_key_mapping(self) -> None:
        if self.mode == "fmri":
            self.key_trigger  = "t"
            self.key_continue = "b"
        else:
            self.key_trigger  = "t"
            self.key_continue = "space"

    # =====================================================================
    #  VISUAL STIMULI
    # =====================================================================

    def _setup_visual_stimuli(self) -> None:
        """Le fixation cross est déjà créé dans BaseTask._init_common_stimuli().
        On ajoute ici le TextStim pour les instructions visuelles inter-blocs."""
        self.cue_stim = visual.TextStim(
            self.win, text="", height=0.06, color="white",
            pos=(0.0, 0.0), wrapWidth=1.5, font="Arial", bold=False,
        )

    # =====================================================================
    #  SOUND SETUP
    # =====================================================================

    def _setup_sounds(self) -> None:
        """Pré-charge tous les stimuli auditifs. Fallback = tons purs."""
        base = Path(self.root_dir) / self.sound_dir
        grasp_path = base / self.grasp_sound_file
        touch_path = base / self.touch_sound_file

        # ── Cue « Grasp » ──
        if grasp_path.exists():
            self.sound_grasp = psysound.Sound(str(grasp_path))
            self.logger.log(f"Loaded grasp sound: {grasp_path}")
        else:
            self.sound_grasp = psysound.Sound(
                value=400, secs=self.cue_duration, stereo=True,
            )
            self.logger.warn(
                f"Grasp WAV not found ({grasp_path}), using 400 Hz tone."
            )

        # ── Cue « Touch » ──
        if touch_path.exists():
            self.sound_touch = psysound.Sound(str(touch_path))
            self.logger.log(f"Loaded touch sound: {touch_path}")
        else:
            self.sound_touch = psysound.Sound(
                value=600, secs=self.cue_duration, stereo=True,
            )
            self.logger.warn(
                f"Touch WAV not found ({touch_path}), using 600 Hz tone."
            )

        # ── Go beep ──
        self.sound_go = psysound.Sound(
            value=self.go_beep_freq, secs=self.go_duration, stereo=True,
        )
        self.logger.log(
            f"Go beep: {self.go_beep_freq} Hz, {self.go_duration} s"
        )

    # =====================================================================
    #  SEQUENCE GENERATION
    # =====================================================================

    @staticmethod
    def _pseudo_random_no_repeat(
        items: List[str], reps: int, max_attempts: int = 500,
    ) -> List[str]:
        """Séquence pseudo-aléatoire sans répétition immédiate."""
        pool = [it for it in items for _ in range(reps)]
        for _ in range(max_attempts):
            seq: List[str] = []
            bag = pool[:]
            random.shuffle(bag)
            ok = True
            while bag:
                candidates = (
                    [x for x in bag if x != seq[-1]] if seq else bag[:]
                )
                if not candidates:
                    ok = False
                    break
                chosen = random.choice(candidates)
                seq.append(chosen)
                bag.remove(chosen)
            if ok and len(seq) == len(pool):
                return seq
        random.shuffle(pool)
        return pool

    def _build_trial_order(self) -> List[str]:
        """20 grasp + 20 touch, pas de répétition immédiate."""
        return self._pseudo_random_no_repeat(
            CONDITIONS, self.n_trials_per_condition,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  TIMELINE CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════

    def _add_event(
        self, onset_s: float, action: str, **kwargs: Any,
    ) -> None:
        """Ajoute un événement à la timeline pré-calculée."""
        event: Dict[str, Any] = {
            "onset_s":   round(onset_s, 6),
            "action":    action,
            "_priority": _ACTION_PRIORITY.get(action, 9),
        }
        event.update(kwargs)
        self.timeline.append(event)

    # ── Construction d'un essai ──────────────────────────────────────────

    def _build_trial_events(
        self,
        t: float,
        trial_idx: int,
        condition: str,
        flip_lead_s: float,
    ) -> float:
        """
        Ajoute tous les événements pour UN essai dans la timeline.

        Chronologie depuis t (trial_start) :
            t + 0                           : trial_start marker + fixation flip
            t + preview_duration            : cue sound (« grasp » / « touch »)
            t + preview_duration + plan_dur : go beep
            t + preview_duration + plan_dur + execute_dur : ITI
            t + preview_duration + plan_dur + execute_dur + iti_dur : trial_end

        Returns
        -------
        float : onset du prochain essai (= trial_end).
        """
        trial_start_t = t

        # Jitter pré-calculé pour ce trial
        jitter = random.uniform(-self.plan_jitter, self.plan_jitter)
        plan_dur = max(2.0, self.plan_duration + jitter)

        # ── TRIAL START ──────────────────────────────────────────────────
        self._add_event(
            t, "marker",
            label="trial_start",
            trial_index=trial_idx,
            condition=condition,
            effector=self.effector,
            plan_duration_s=round(plan_dur, 3),
        )
        if self.parport_actif:
            self._add_event(
                t, "parport_event",
                label="trial_start_trigger",
                trial_index=trial_idx,
                condition=condition,
                pin_code=self.event_codes.get("trial_start", 1),
            )

        # ── PREVIEW PHASE : fixation (flip lead avant le premier son) ──
        # Le flip est 2 frames AVANT le premier son pour ne pas bloquer.
        # Si preview_dur > flip_lead, on planifie le flip à t ;
        # le prochain son est à t + preview_dur — largement suffisant.
        self._add_event(
            t, "visual_fixation",
            label="preview_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t + 0.001, "parport_event",
                label="preview_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes.get("preview_start", 2),
            )

        t_cue = trial_start_t + self.preview_duration

        # ── AUDITORY CUE (« Grasp » / « Touch ») → début Plan ───────────
        cue_code_key = f"cue_{condition}"
        self._add_event(
            t_cue, "sound_cue",
            label=f"cue_{condition}",
            trial_index=trial_idx,
            condition=condition,
            sound_id=condition,
            pin_code=self.event_codes.get(cue_code_key, 10),
        )

        # ── GO BEEP → début Execute ─────────────────────────────────────
        t_go = t_cue + plan_dur
        go_code_key = f"go_{condition}"
        self._add_event(
            t_go, "sound_go",
            label="go_beep",
            trial_index=trial_idx,
            condition=condition,
            pin_code=self.event_codes.get(go_code_key, 30),
        )
        self._add_event(
            t_go + 0.001, "marker",
            label="execute_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t_go + 0.002, "parport_event",
                label="execute_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes.get("execute_start", 50),
            )

        # ── ITI ─────────────────────────────────────────────────────────
        t_iti = t_go + self.execute_duration

        self._add_event(
            t_iti, "marker",
            label="iti_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t_iti, "parport_event",
                label="iti_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes.get("iti_start", 60),
            )
        # Refresh de l'écran (fixation) pour maintenir l'affichage
        self._add_event(
            t_iti + 0.003, "visual_fixation",
            label="iti_fixation",
            trial_index=trial_idx,
        )

        # ── TRIAL END ───────────────────────────────────────────────────
        t_end = t_iti + self.iti_duration

        self._add_event(
            t_end, "marker",
            label="trial_end",
            trial_index=trial_idx,
            condition=condition,
            trial_duration_s=round(t_end - trial_start_t, 3),
        )

        return t_end

    # ── Construction complète de la timeline ─────────────────────────────

    def _build_full_timeline(self) -> None:
        """
        Construit la timeline complète AVANT le trigger de démarrage.
        """
        self.timeline.clear()

        trial_order = self._build_trial_order()
        t = 0.0

        flip_lead_s = 2.0 * self.frame_duration_s

        # ── Run start ────────────────────────────────────────────────────
        self._add_event(
            t, "marker", label="run_start",
            effector=self.effector,
            run_number=self.run_number,
            n_trials=len(trial_order),
            trial_order=str(trial_order),
        )
        if self.parport_actif:
            self._add_event(
                t + 0.001, "parport_event",
                label="run_start_trigger",
                pin_code=self.event_codes.get("run_start", 100),
            )

        # ── Baseline initiale : fixation ─────────────────────────────────
        self._add_event(t, "visual_fixation", label="baseline_start")
        t += self.initial_baseline

        # ── Essais ───────────────────────────────────────────────────────
        for trial_idx, condition in enumerate(trial_order, start=1):
            t = self._build_trial_events(
                t, trial_idx, condition, flip_lead_s,
            )

        # ── Baseline finale ──────────────────────────────────────────────
        self._add_event(t, "visual_fixation", label="final_baseline_start")
        t += self.final_baseline

        # ── Run end ──────────────────────────────────────────────────────
        self._add_event(t, "marker", label="run_end")
        if self.parport_actif:
            self._add_event(
                t, "parport_event",
                label="run_end_trigger",
                pin_code=self.event_codes.get("run_end", 200),
            )

        # ── Tri stable : onset, puis priorité ────────────────────────────
        self.timeline.sort(key=lambda e: (e["onset_s"], e["_priority"]))

        # Numérotation séquentielle
        for i, evt in enumerate(self.timeline):
            evt["event_index"] = i

        # Validation
        self._validate_no_flip_sound_collision()

        # Résumé
        n_sound = sum(
            1 for e in self.timeline if e["action"].startswith("sound_")
        )
        n_vis = sum(
            1 for e in self.timeline if e["action"].startswith("visual_")
        )
        n_pp = sum(
            1 for e in self.timeline if e["action"] == "parport_event"
        )
        n_mark = len(self.timeline) - n_sound - n_vis - n_pp
        total_dur = self.timeline[-1]["onset_s"] if self.timeline else 0

        self.logger.log(
            f"Timeline built: {len(self.timeline)} events "
            f"({n_sound} sound, {n_vis} visual, {n_pp} parport, "
            f"{n_mark} markers) | "
            f"~{total_dur:.1f} s ({total_dur / 60:.1f} min)"
        )

    def _validate_no_flip_sound_collision(self) -> None:
        """
        Vérifie qu'aucun flip bloquant n'est planifié au même onset
        qu'un événement sonore critique.
        """
        visual_onsets = set()
        sound_onsets  = set()

        for evt in self.timeline:
            if evt["action"].startswith("visual_"):
                visual_onsets.add(evt["onset_s"])
            elif evt["action"].startswith("sound_"):
                sound_onsets.add(evt["onset_s"])

        collisions = visual_onsets & sound_onsets
        if collisions:
            self.logger.err(
                f"TIMELINE BUG: {len(collisions)} flip/sound collisions "
                f"detected! First at t={min(collisions):.3f} s. "
                f"Flips could delay sound onsets."
            )
        else:
            self.logger.ok(
                "Timeline validated: no flip/sound collisions."
            )

    # ── Sauvegarde planned ───────────────────────────────────────────────

    def _save_planned_timeline(self) -> None:
        """Sauvegarde la timeline pré-calculée avant exécution."""
        if not self.enregistrer or not self.timeline:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.task_name.replace(' ', '')
        fname = (
            f"{self.nom}_{safe_name}"
            f"_{self.effector}_run{self.run_number:02d}"
            f"_{timestamp}_planned.csv"
        )
        path = os.path.join(self.data_dir, fname)
        try:
            all_keys = sorted(
                set().union(*(e.keys() for e in self.timeline))
                - {"_priority"}
            )
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=all_keys, extrasaction='ignore',
                )
                writer.writeheader()
                for evt in self.timeline:
                    writer.writerow(
                        {k: v for k, v in evt.items() if k != "_priority"}
                    )
            self.logger.ok(f"Planned timeline saved: {path}")
        except Exception as e:
            self.logger.err(f"Failed to save planned timeline: {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  TIMELINE EXECUTION — Moteur temps-réel minimal
    # ═════════════════════════════════════════════════════════════════════

    def _wait_until(
        self, target_s: float, high_precision: bool = False,
    ) -> None:
        """
        Attend jusqu'à target_s sur task_clock.

        high_precision=True (sons) :
            Sleep pour le gros, spin-wait pour les 2 dernières ms.
        high_precision=False (visuels, marqueurs) :
            core.wait standard (suffisant).
        """
        remaining = target_s - self.task_clock.getTime()
        if remaining <= 0:
            return

        if high_precision:
            if remaining > 0.003:
                core.wait(remaining - 0.002, hogCPUperiod=0.0)
            while self.task_clock.getTime() < target_s:
                pass
        else:
            core.wait(remaining, hogCPUperiod=0.0)

    def _dispatch_event(self, event: Dict[str, Any]) -> float:
        """
        Exécute l'action d'un événement. Retourne le temps réel.

        Pour les sons : trigger EEG d'abord (µs), puis play() (~0.5 ms).
        Le marqueur EEG est donc aligné au mieux avec l'onset sonore.
        """
        action = event["action"]

        # ── Visuels ──────────────────────────────────────────────────────
        if action == "visual_fixation":
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "visual_instruction":
            self.cue_stim.text = event.get("instruction_text", "")
            self.cue_stim.draw()
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        # ── Son : cue (« Grasp » / « Touch ») ───────────────────────────
        if action == "sound_cue":
            # 1) Trigger EEG (le plus rapide, ~µs)
            if self.parport_actif:
                self.ParPort.send_trigger(event.get("pin_code", 0))
            # 2) Play son
            snd = (
                self.sound_grasp
                if event.get("sound_id") == "grasp"
                else self.sound_touch
            )
            snd.stop()   # reset si encore en lecture
            snd.play()
            return self.task_clock.getTime()

        # ── Son : go beep ────────────────────────────────────────────────
        if action == "sound_go":
            if self.parport_actif:
                self.ParPort.send_trigger(event.get("pin_code", 0))
            self.sound_go.stop()
            self.sound_go.play()
            return self.task_clock.getTime()

        # ── Trigger EEG seul ─────────────────────────────────────────────
        if action == "parport_event":
            if self.parport_actif:
                self.ParPort.send_trigger(event.get("pin_code", 0))
            return self.task_clock.getTime()

        # ── Marqueur logique (pas d'action hardware) ─────────────────────
        if action == "marker":
            return self.task_clock.getTime()

        return self.task_clock.getTime()

    # ── Record ───────────────────────────────────────────────────────────

    def _build_execution_record(
        self, event: Dict[str, Any], actual_time_s: float,
    ) -> Dict[str, Any]:
        """Construit l'enregistrement pour UN événement exécuté."""
        error_ms = (actual_time_s - event["onset_s"]) * 1000.0
        return {
            "participant":         self.nom,
            "session":             self.session,
            "effector":            self.effector,
            "run_number":          self.run_number,
            "event_index":         event.get("event_index", ""),
            "action":              event["action"],
            "label":               event.get("label", ""),
            "onset_planned_s":     event["onset_s"],
            "onset_actual_s":      round(actual_time_s, 6),
            "scheduling_error_ms": round(error_ms, 3),
            "trial_index":         event.get("trial_index", ""),
            "condition":           event.get("condition", ""),
            "sound_id":            event.get("sound_id", ""),
            "pin_code":            event.get("pin_code", ""),
            "plan_duration_s":     event.get("plan_duration_s", ""),
            "trial_duration_s":    event.get("trial_duration_s", ""),
        }

    # ── Boucle principale ────────────────────────────────────────────────

    def _execute_timeline(self) -> None:
        """
        Parcourt la timeline pré-calculée. Seule boucle active pendant
        l'acquisition EEG.
        """
        n_events = len(self.timeline)
        self.logger.log(f"Executing timeline: {n_events} events …")

        # ══ GC désactivé pour tout le run ══
        gc.disable()

        try:
            for i, event in enumerate(self.timeline):

                # ── Quit check sur événements non-critiques ──
                is_sound = event["action"].startswith("sound_")
                if not is_sound:
                    self.should_quit()

                # ── Attente de l'onset ──
                self._wait_until(
                    event["onset_s"], high_precision=is_sound,
                )

                # ── Exécution ──
                actual_t = self._dispatch_event(event)

                # ── Enregistrement ──
                record = self._build_execution_record(event, actual_t)
                self.global_records.append(record)
                self.save_trial_incremental(record)

                # ── Eyetracker ──
                if self.eyetracker_actif:
                    label = event.get("label", event["action"])
                    self.EyeTracker.send_message(
                        f"R{self.run_number:02d}_"
                        f"E{i:04d}_"
                        f"{label.upper()}"
                    )

                # ── Alerte timing (sons uniquement, seuil > 1 ms) ──
                if is_sound:
                    err_ms = record["scheduling_error_ms"]
                    if abs(err_ms) > 1.0:
                        self.logger.warn(
                            f"TIMING E{i} "
                            f"T{event.get('trial_index', '?')} "
                            f"({event.get('label', '?')}): "
                            f"{err_ms:+.2f} ms"
                        )

                # ── Log de progression ──
                if event.get("label") == "trial_end":
                    t_idx = event.get("trial_index", "?")
                    cond  = event.get("condition", "?")
                    dur   = event.get("trial_duration_s", "?")
                    self.logger.log(
                        f"  Trial {t_idx}/{self.n_trials_total} "
                        f"({cond}) done  "
                        f"[t={actual_t:.2f} s, dur={dur} s]"
                    )

        finally:
            gc.enable()
            gc.collect()

        self.logger.ok("Timeline execution complete.")

        # ── Résumé timing ──
        sound_records = [
            r for r in self.global_records
            if r["action"].startswith("sound_")
            and r["scheduling_error_ms"] != ""
        ]
        if sound_records:
            errors    = [abs(r["scheduling_error_ms"]) for r in sound_records]
            mean_err  = sum(errors) / len(errors)
            max_err   = max(errors)
            n_over_05 = sum(1 for e in errors if e > 0.5)
            n_over_1  = sum(1 for e in errors if e > 1.0)
            n_over_2  = sum(1 for e in errors if e > 2.0)
            self.logger.log(
                f"Timing summary: {len(sound_records)} sound events | "
                f"mean |err| = {mean_err:.3f} ms | "
                f"max |err| = {max_err:.3f} ms | "
                f">0.5 ms: {n_over_05} | "
                f">1 ms: {n_over_1} | >2 ms: {n_over_2}"
            )

    # ═════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY
    # ═════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        finished = False

        try:
            self._show_instructions()
            self.wait_for_trigger()

            self._execute_timeline()

            finished = True
            self.logger.ok(
                f"Run {self.run_number:02d} ({self.effector}) done."
            )

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Interruption manuelle.")

        except Exception as exc:
            self.logger.err(f"CRITICAL: {exc}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Arrêt sécurisé des sons
            for snd in (self.sound_grasp, self.sound_touch, self.sound_go):
                try:
                    snd.stop()
                except Exception:
                    pass

            if self.eyetracker_actif:
                self.EyeTracker.stop_recording()
                self.EyeTracker.send_message("END_EXP")
                self.EyeTracker.close_and_transfer_data(self.data_dir)

            saved_path = self.save_data(
                data_list=self.global_records,
                filename_suffix=(
                    f"_{self.effector}_run{self.run_number:02d}"
                ),
            )

            if saved_path and os.path.exists(saved_path):
                try:
                    from tasks.qc.qc_motor_planning import qc_motor_planning
                    qc_motor_planning(saved_path)
                except ImportError:
                    self.logger.warn("QC module not found (non bloquant)")
                except Exception as qc_exc:
                    self.logger.warn(
                        f"QC échoué (non bloquant) : {qc_exc}"
                    )

            if finished:
                self.show_instructions(
                    f"Run {self.run_number:02d} terminé.\n"
                    f"Effecteur : {self.effector}\nMerci !"
                )
                core.wait(3.0)

    # ─────────────────────────────────────────────────────────────────────
    #  INSTRUCTIONS
    # ─────────────────────────────────────────────────────────────────────

    def _show_instructions(self) -> None:
        effector_label = "la MAIN" if self.effector == "hand" else "l'OUTIL"
        txt = (
            f"PLANIFICATION MOTRICE — Run {self.run_number:02d}\n"
            f"Effecteur : {effector_label}\n\n"
            "À chaque essai :\n"
            "  1. Fixez la croix sur la table\n"
            "  2. Écoutez l'instruction audio\n"
            "     • « Grasp » → Saisissez l'objet\n"
            "     • « Touch » → Touchez l'objet\n"
            "  3. Attendez le BIP pour exécuter le mouvement\n"
            "  4. Revenez à la position de départ\n\n"
            "Maintenez votre regard sur la croix de fixation.\n\n"
            "En attente du démarrage …"
        )
        self.show_instructions(txt)