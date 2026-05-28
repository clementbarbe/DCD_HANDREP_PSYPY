Hand Representation Task

Tâche de représentation implicite de la main basée sur le protocole de Longo & Haggard (2010).
Conçue pour l'évaluation de la représentation corporelle — CENIR, Institut du Cerveau (ICM).

> **Référence** : Longo, M. R., & Haggard, P. (2010). An implicit body representation underlying human position sense.
> *Proceedings of the National Academy of Sciences*, 107(26), 11727–11732.
> [doi:10.1073/pnas.1003483107](https://pubmed.ncbi.nlm.nih.gov/20547858/)

## Principe

1. Une image indiquant un **doigt** et une **zone** cible apparaît à l'écran
2. Le participant pointe la zone indiquée sur sa propre main **avant la fin d'une barre de progression** (4 s)
3. Une **photo webcam** est capturée automatiquement à la fin du délai
4. Le participant revoit un écran de repos (2 s) avant l'essai suivant
5. Des **photos de référence** (main au repos) sont prises en début et fin de bloc

Les images sources représentent une **main gauche**. Pour la main droite, un miroir horizontal
est appliqué automatiquement, et les noms de doigts sont inversés anatomiquement
(pouce ↔ auriculaire, index ↔ annulaire, majeur inchangé).

## Structure

```scss
Block (×1 par session)
├── REF1 — photo de référence initiale
├── 10 miniblocs × 10 positions = 100 essais
│   └── Essai
│       ├── Affichage image + barre de progression (4 s)
│       ├── Capture webcam
│       └── Retour à la position de base (2 s)
└── REF2 — photo de référence finale
```
## Positions

Chaque essai cible l'une des 10 positions (5 doigts × 2 zones) :

| Doigt | Zone 1 | Zone 2 |
|-------|--------|--------|
| Pouce | a2.png | a1.png |
| Index | a4.png | a3.png |
| Majeur | a6.png | a5.png |
| Annulaire | a8.png | a7.png |
| Auriculaire | a10.png | a9.png |

## Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `hand` | `droite` | Main testée (`droite` ou `gauche`) |
| `n_blocks` | `1` | Nombre de blocs (100 essais chacun) |
| `trial_duration` | `4.0` | Durée d'affichage de l'image (secondes) |
| `camera_index` | auto | Index webcam (max détecté automatiquement) |
| `session` | `01` | Numéro de session |

## Calibration caméra

Un module de calibration indépendant permet d'ajuster le cadrage de la webcam
avant la passation. Un point rouge central est affiché sur le flux live pour
aligner la caméra. La capture confirmée est sauvegardée en PNG 1920×1080 **sans overlay**.

| Surface | Sortie |
|---------|--------|
| Table | `data/calib/table_YYYYMMDD_HHMMSS.png` |
| Plateau | `data/calib/plateau_YYYYMMDD_HHMMSS.png` |

## Lancement

### Prérequis

- Python 3.10+
- PsychoPy 2025.1.1
- OpenCV (`opencv-python`)
- PyQt6

### Démarrage

```bash
python main.py
```
Ou via le fichier Launch.bat sous Windows.

L'interface graphique permet de sélectionner la main, le bloc et de lancer
la calibration ou la tâche.
Arborescence

```scss
projet/
├── main.py                          ← point d'entrée
├── Launch.bat                       ← raccourci Windows
├── images/                          ← images des positions (a1.png … a10.png)
├── gui/
│   ├── menu.py                      ← fenêtre principale PyQt6
│   └── tabs/
│       ├── tabs_hand.py             ← onglet Hand Representation
│       └── tabs_calibration.py      ← onglet Calibration caméra
├── tasks/
│   ├── hand_representation.py       ← tâche principale (PsychoPy)
│   ├── camera_calibration.py        ← orchestrateur calibration
│   └── _calibration_ui.py           ← UI calibration (subprocess PyQt6)
└── data/
    ├── calib/                       ← captures de calibration
    │   ├── table_20250614_153022.png
    │   └── plateau_20250614_153134.png
    └── handrep/
        └── ID_Participant/          ← données par participant
            ├── *_final_*.csv        ← résultats complets
            ├── *_incremental.csv    ← backup trial par trial
            └── photos/
                ├── S1_REF1.jpg      ← référence initiale
                ├── S1_T1_ring_Z2.jpg
                ├── S1_T2_thumb_Z1.jpg
                ├── …
                └── S1_REF2.jpg      ← référence finale
```
## Données

### Nommage des photos

| Type | Format | Exemple |
|-------|--------|--------|
| Essai | S{session}_T{trial}_{doigt}_Z{zone}.jpg	S1_T6_ring_Z2.jpg | S1_T6_ring_Z2.jpg |
| Référence | S{session}_REF{1 ou 2}.jpg | S1_REF1.jpg |

### Métriques enregistrées (CSV)

    Identifiants : participant, session, hand, block_number, miniblock_number, global_trial
    Position : finger_source, finger_displayed, zone, image_file
    Timing : image_onset, capture_time_task, trial_duration
    Fichiers : photo_filename, photo_path
    Métadonnées : flip_horiz, wall_timestamp

## Sauvegarde

Deux fichiers CSV sont produits pour chaque session :
Fichier	Écriture	Rôle
*_incremental.csv	Après chaque essai (append)	Protection anti-crash
*_final_*.csv	En fin de session	Fichier propre et complet

Un mécanisme de sauvegarde d'urgence est déclenché automatiquement
en cas d'interruption (Échap / Ctrl+C).
Auteur

Clément BARBE — CENIR, Institut du Cerveau (ICM), Paris