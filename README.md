

## 📁 Project Files

```
detection/
├── emotion_detector.py            ← Main script (uses FER)
├── emotion_detector_deepface.py   ← Alternative script (uses DeepFace)
├── test_fer.py                    ← Diagnostic test script
├── requirements.txt               ← All dependencies
├── README.md                      ← This guide
├── venv/                          ← Your Python 3.10 virtual environment
├── screenshots/                   ← Saved screenshots (auto-created)
└── logs/                          ← Detection logs (auto-created)
```

---

## ⚙️ System Requirements

| Requirement | Value |
|-------------|-------|
| Python | **3.10.x only** (NOT 3.11, 3.12, or 3.14) |
| NumPy | **Must be below 2.0** (numpy<2) |
| OS | Windows 10 / 11 |
| Camera | Any built-in or USB webcam |
| RAM | 4 GB minimum, 8 GB recommended |

---

## 🚀 Full Setup Guide (Windows)

### Step 1 — Create virtual environment with Python 3.10

Open Command Prompt in your project folder:

```
py -3.10 -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of the prompt.

---

### Step 2 — If activation is blocked (PowerShell error)

Switch to Command Prompt instead of PowerShell, OR run:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again:
```
venv\Scripts\activate
```

---

### Step 3 — Install packages in this exact order

```
python -m pip install --upgrade pip
pip install "numpy<2"
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install fer==22.5.1
pip install opencv-python==4.8.1.78
```

---

### Step 4 — Verify everything works

```
python -c "import cv2, tensorflow as tf, numpy as np; from fer import FER; print('ALL OK | TF:', tf.__version__, '| NumPy:', np.__version__, '| CV2:', cv2.__version__)"
```

Expected output:
```
ALL OK | TF: 2.15.0 | NumPy: 1.26.x | CV2: 4.8.1
```

---

### Step 5 — Run the detector

```
python emotion_detector.py
```

Camera window opens and emotions appear on your face in real time.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save screenshot to `screenshots/` folder |
| `B` | Toggle emotion bar chart on/off |

---

## 🖥️ Command Line Options

```
# Default — built-in webcam
python emotion_detector.py

# External USB camera
python emotion_detector.py --camera 1

# Process every frame (most accurate, slower)
python emotion_detector.py --skip 1

# Better face detection (needs: pip install mtcnn)
python emotion_detector.py --mtcnn

# Analyse a photo instead of camera
python emotion_detector.py --image photo.jpg
```

---

## 🔄 Option B — DeepFace Version (if FER fails)

If FER does not work, use the DeepFace version instead:

```
pip install deepface
python emotion_detector_deepface.py
```

> ⚠️ First launch downloads model files automatically (takes 1-2 minutes).
> After that it runs instantly.

DeepFace options:
```
python emotion_detector_deepface.py --camera 0
python emotion_detector_deepface.py --skip 1
```

---

## ❌ Troubleshooting

### Problem: NumPy 2.x error — `_ARRAY_API not found`
```
AttributeError: _ARRAY_API not found
```
**Fix:**
```
pip install "numpy<2"
```

---

### Problem: FER import error
```
ImportError: cannot import name 'FER' from 'fer'
```
**Fix:**
```
pip uninstall fer -y
pip install fer==22.5.1
pip install keras==2.15.0
```
If it still fails, switch to DeepFace (Option B above).

---

### Problem: OpenCV missing CascadeClassifier
```
AttributeError: module 'cv2' has no attribute 'CascadeClassifier'
```
**Fix:**
```
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-python==4.8.1.78
```

---

### Problem: Camera does not open
```
[ERROR] Cannot open camera 0
```
**Fix on Windows:**
- Settings → Privacy → Camera → turn ON "Allow apps to access your camera"
- Try a different camera index:
```
python emotion_detector.py --camera 1
python emotion_detector.py --camera 2
```

---

### Problem: PowerShell blocks venv activation
```
cannot be loaded because running scripts is disabled
```
**Fix — use Command Prompt instead of PowerShell, OR run:**
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Problem: TensorFlow not found for Python version
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.13.0
```
**Fix — use 2.15.0 instead:**
```
pip install tensorflow==2.15.0
```

---

### Problem: Face detected but no emotions showing
- Make sure room is well lit (face a lamp or window)
- Sit 40–70 cm from camera
- Look directly at the camera
- Try processing every frame:
```
python emotion_detector.py --skip 1
```
- Run diagnostic test:
```
python test_fer.py
```

---

### Problem: Slow FPS / laggy
```
python emotion_detector.py --skip 3
python emotion_detector.py --skip 4
```

---

## 🛠️ Full Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Language | Python | 3.10.x |
| Camera feed | OpenCV | 4.8.1.78 |
| Face detection | OpenCV Haar + FER | — |
| Emotion model | FER or DeepFace | 22.5.1 / latest |
| Deep learning | TensorFlow | 2.15.0 |
| Array processing | NumPy | < 2.0 |
| Visualisation | OpenCV drawing | — |

---

## 🔁 Every Time You Open a New Terminal

You must activate the venv before running anything:

```
cd C:\Users\Sami Nadeem\OneDrive\Desktop\detection
venv\Scripts\activate
python emotion_detector.py
```

If you skip `venv\Scripts\activate` the wrong Python will run and nothing will work.

---

## 📊 What the Screen Shows

```
┌─────────────────────────┐
│ FACE EMOTION DETECTOR   │  ← Info panel (top-left corner)
│ Time    : 14:32:07      │
│ FPS     : 26.4          │
│ Faces   : 1             │
│ Session : 8s            │
│ Emotion : Happy  :)     │
│ Q=Quit  S=Screenshot    │
└─────────────────────────┘

  L┐ Happy :)  92% ┌L       ← Emotion badge above face
   │               │         ← Bounding box around face
   │      #1       │
  L┘               └L        ← Corner accents

                              happy     ████████░  92%
                              neutral   █░░░░░░░░   5%  ← Bars (B to hide)
                              sad       ░░░░░░░░░   2%
```

---

*Ello Face Emotion Detection System — Built for Teyzix Core Internship*
