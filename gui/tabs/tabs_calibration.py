# tabs/tabs_calibration.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QCheckBox,
)


class CameraCalibrationTab(QWidget):
    """Tab for standalone camera calibration (table and/or plateau).
    
    Camera index is auto-detected (max index) — no manual selector.
    """

    SESSION_MAP = {
        "Session 1 (Pre)":  "1",
        "Session 2 (Pre)":  "2",
        "Session 3 (Post)": "3",
    }

    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self._build_session_group())
        layout.addWidget(self._build_surfaces_group())
        layout.addWidget(self._build_options_group())
        layout.addWidget(self._build_launch_group())
        layout.addStretch()

    def _build_session_group(self):
        group = QGroupBox("Session")
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Session :"))
        self.combo_session = QComboBox()
        self.combo_session.addItems(list(self.SESSION_MAP.keys()))
        hbox.addWidget(self.combo_session)
        hbox.addStretch()
        group.setLayout(hbox)
        return group

    def _build_surfaces_group(self):
        group = QGroupBox("Surfaces à calibrer")
        vbox = QVBoxLayout()
        self.check_table = QCheckBox("Table")
        self.check_plateau = QCheckBox("Plateau")
        self.check_table.setChecked(True)
        self.check_plateau.setChecked(True)
        vbox.addWidget(self.check_table)
        vbox.addWidget(self.check_plateau)
        group.setLayout(vbox)
        return group

    def _build_options_group(self):
        group = QGroupBox("Options")
        vbox = QVBoxLayout()
        self.check_flip = QCheckBox(
            "Retourner le flux verticalement (si image à l'envers)"
        )
        vbox.addWidget(self.check_flip)
        group.setLayout(vbox)
        return group

    def _build_launch_group(self):
        group = QGroupBox("Lancement")
        vbox = QVBoxLayout()
        btn = QPushButton("Lancer la calibration")
        btn.clicked.connect(self.run_task)
        vbox.addWidget(btn)
        group.setLayout(vbox)
        return group

    def _calibration_types(self):
        types = []
        if self.check_table.isChecked():
            types.append("table")
        if self.check_plateau.isChecked():
            types.append("plateau")
        return types

    def get_common(self):
        return {
            "tache": "CameraCalibration",
            "session": self.SESSION_MAP[self.combo_session.currentText()],
            "calibration_types": self._calibration_types(),
            "flip_feed": self.check_flip.isChecked(),
            "camera_index": -1,  # auto-detect max
        }

    def run_task(self):
        self.parent_menu.run_experiment(self.get_common())