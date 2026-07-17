

import cv2
import sys
import time
import argparse
import datetime
import os
import numpy as np

# ── FER import with friendly error ────────────────────────────────────────────
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# BGR colours for each emotion label
EMOTION_COLORS = {
    "happy":    (0,   220,  90),   # green
    "sad":      (255, 140,  60),   # orange-blue
    "angry":    (50,   50, 255),   # red
    "fear":     (180,  30, 255),   # purple
    "surprise": (0,   220, 255),   # yellow
    "disgust":  (30,  160,  60),   # dark green
    "neutral":  (160, 160, 160),   # grey
}

# Human-readable labels shown on screen
EMOTION_LABELS = {
    "happy":    "Happy 😊",
    "sad":      "Sad 😢",
    "angry":    "Angry 😠",
    "fear":     "Fear 😨",
    "surprise": "Surprise 😮",
    "disgust":  "Disgust 🤢",
    "neutral":  "Neutral 😐",
}

SCREENSHOT_DIR = "screenshots"
LOG_DIR        = "logs"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_rounded_box(frame, x1, y1, x2, y2, color, thickness=2, r=14):
    """
    Draw a rounded-corner rectangle — looks much cleaner than cv2.rectangle.
    Draws 4 lines and 4 quarter-circle arcs to form the corners.
    """
    cv2.line(frame, (x1 + r, y1),     (x2 - r, y1),     color, thickness)
    cv2.line(frame, (x1 + r, y2),     (x2 - r, y2),     color, thickness)
    cv2.line(frame, (x1,     y1 + r), (x1,     y2 - r), color, thickness)
    cv2.line(frame, (x2,     y1 + r), (x2,     y2 - r), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180,  0,  90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270,  0,  90, color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r),  90,  0,  90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r),   0,  0,  90, color, thickness)


def draw_corner_accents(frame, x1, y1, x2, y2, color, length=20, thickness=3):
    """
    Draw L-shaped corner accents inside/outside the box for a techy look.
    Only draws the first 'length' pixels of each corner edge.
    """
    tl = [(x1, y1 + length), (x1, y1), (x1 + length, y1)]
    tr = [(x2 - length, y1), (x2, y1), (x2, y1 + length)]
    bl = [(x1, y2 - length), (x1, y2), (x1 + length, y2)]
    br = [(x2 - length, y2), (x2, y2), (x2, y2 - length)]
    for pts in (tl, tr, bl, br):
        cv2.polylines(frame, [np.array(pts)], False, color, thickness)


def draw_label_badge(frame, text, x, y, color, font_scale=0.65, thickness=2):
    """
    Draw a filled pill-shaped badge containing 'text' at position (x, y).
    Returns the badge height so callers can position other elements.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 10, 6
    bx1 = x
    by1 = y - th - pad_y * 2
    bx2 = x + tw + pad_x * 2
    by2 = y

    # Filled background
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    # Text in black for readability on any badge colour
    cv2.putText(frame, text, (bx1 + pad_x, by2 - pad_y),
                font, font_scale, (0, 0, 0), thickness)
    return th + pad_y * 2


def draw_emotion_bars(frame, emotions: dict, x: int, y: int, bar_w=160):
    """
    Draw a compact vertical bar-chart of all emotion probabilities.
    Sorted highest to lowest, with coloured bars and percentage labels.
    """
    if not emotions:
        return

    bar_h   = 13
    spacing = 4
    total_h = len(emotions) * (bar_h + spacing) + 10

    # Semi-transparent dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 4),
                  (x + bar_w + 90, y + total_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    sorted_emos = sorted(emotions.items(), key=lambda e: e[1], reverse=True)

    for i, (emo, score) in enumerate(sorted_emos):
        row_y  = y + i * (bar_h + spacing)
        filled = int(score * bar_w)
        color  = EMOTION_COLORS.get(emo, (160, 160, 160))

        # Grey background track
        cv2.rectangle(frame, (x, row_y), (x + bar_w, row_y + bar_h),
                      (50, 50, 50), -1)
        # Coloured fill
        if filled > 0:
            cv2.rectangle(frame, (x, row_y), (x + filled, row_y + bar_h),
                          color, -1)
        # Label: "happy   87%"
        label = f"{emo:<8} {score * 100:4.0f}%"
        cv2.putText(frame, label, (x + bar_w + 6, row_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)


def draw_hud(frame, fps: float, face_count: int,
             top_emotion: str, elapsed: float):
    """
    Draw the top-left HUD panel showing system status.
    Uses a semi-transparent dark background for legibility.
    """
    panel_w, panel_h = 295, 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Green border on HUD
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h),
                  (0, 220, 90), 1)

    color = EMOTION_COLORS.get(top_emotion, (160, 160, 160))
    label = EMOTION_LABELS.get(top_emotion, top_emotion.upper())
    ts    = datetime.datetime.now().strftime("%H:%M:%S")

    rows = [
        ("FACE EMOTION SYSTEM",           (0, 220, 90),  0.45, 1),
        (f"Time    : {ts}",               (200,200,200), 0.38, 1),
        (f"FPS     : {fps:.1f}",          (200,200,200), 0.38, 1),
        (f"Faces   : {face_count}",       (200,200,200), 0.38, 1),
        (f"Session : {int(elapsed)}s",    (200,200,200), 0.38, 1),
        (f"Emotion : {label}",            color,         0.42, 1),
        ("Q = Quit  S = Screenshot",      (80, 80, 80),  0.33, 1),
    ]
    for i, (text, col, scale, thick) in enumerate(rows):
        cv2.putText(frame, text, (18, 30 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)


# ══════════════════════════════════════════════════════════════════════════════
# FACE ANALYSER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FaceAnalyzer:
    """
    Handles all face detection and emotion classification logic.

    Parameters
    ----------
    camera_index : int   — which webcam to open (0 = built-in)
    skip_frames  : int   — only run emotion detection every N frames
                          (improves FPS on slow machines; display still shows every frame)
    use_mtcnn    : bool  — use MTCNN face detector inside FER (slower but more accurate)
    show_bars    : bool  — show emotion probability bar chart
    """

    def __init__(self,
                 camera_index: int  = 0,
                 skip_frames:  int  = 2,
                 use_mtcnn:    bool = False,
                 show_bars:    bool = True):

        self.camera_index = camera_index
        self.skip_frames  = skip_frames
        self.show_bars    = show_bars
        self.fer          = None
        self.haar         = None

        self._load_models(use_mtcnn)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self, use_mtcnn: bool):
        """Load FER emotion model and OpenCV Haar cascade."""

        # FER model (TensorFlow-backed deep learning classifier)
        if FER_AVAILABLE:
            print("[INFO] Loading FER emotion model …")
            try:
                self.fer = FER(mtcnn=use_mtcnn)
                print("[INFO] FER loaded successfully ✅")
            except Exception as e:
                print(f"[WARN] FER failed: {e}")
                print("[INFO] Falling back to Haar cascade only.")

        # Haar cascade — fast frontal face detector (used as fallback)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar    = cv2.CascadeClassifier(cascade_path)
        if self.haar.empty():
            print("[ERROR] Haar cascade file not found. Reinstall OpenCV.")
            sys.exit(1)
        print("[INFO] Haar cascade loaded ✅")

    # ── Frame downscale helper ────────────────────────────────────────────────

    @staticmethod
    def _downscale(frame, scale: float = 0.5):
        """
        Shrink frame for faster detection, return scaled frame + ratio.
        Detection runs on the small frame; boxes are scaled back up for display.
        """
        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return small, 1.0 / scale

    # ── Haar fallback detector ────────────────────────────────────────────────

    def _detect_haar(self, gray):
        """Detect faces using Haar cascade (no emotion info)."""
        return self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    # ── Annotate one face ─────────────────────────────────────────────────────

    def _annotate_face(self, frame, x, y, w, h,
                       emotions: dict = None, face_id: int = 1):
        """
        Draw bounding box, corner accents, emotion badge, and bar chart
        for a single detected face.
        """
        x2, y2 = x + w, y + h

        # Pick colour based on top emotion (grey if unknown)
        if emotions:
            top_emo  = max(emotions, key=emotions.get)
            top_conf = emotions[top_emo]
            color    = EMOTION_COLORS.get(top_emo, (160, 160, 160))
        else:
            top_emo  = "unknown"
            top_conf = 0.0
            color    = (160, 160, 160)

        # Rounded bounding box
        draw_rounded_box(frame, x, y, x2, y2, color, thickness=2)

        # Corner accent marks (techy L-shapes)
        draw_corner_accents(frame, x, y, x2, y2, color, length=18, thickness=2)

        # Subtle glow (semi-transparent thick border)
        glow = frame.copy()
        cv2.rectangle(glow, (x - 3, y - 3), (x2 + 3, y2 + 3), color, 8)
        cv2.addWeighted(glow, 0.25, frame, 0.75, 0, frame)

        # Emotion label badge above the box
        if emotions:
            label = f"{EMOTION_LABELS.get(top_emo, top_emo)}  {top_conf * 100:.0f}%"
            badge_y = max(y - 2, 30)
            draw_label_badge(frame, label, x, badge_y, color)

            # Face ID dot + number (top-right of box)
            cv2.circle(frame, (x2 - 10, y + 12), 5, color, -1)
            cv2.putText(frame, f"#{face_id}",
                        (x2 - 34, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # Emotion probability bars (right of box, or left if near edge)
            if self.show_bars:
                bar_x = x2 + 10
                if bar_x + 260 > frame.shape[1]:
                    bar_x = max(x - 264, 0)
                draw_emotion_bars(frame, emotions, bar_x, y)
        else:
            # Haar fallback — just show "Face Detected"
            cv2.putText(frame, "Face Detected",
                        (x, max(y - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, "Install FER for emotions",
                        (x, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 180, 255), 1)

    # ── Main camera loop ──────────────────────────────────────────────────────

    def run(self):
        """
        Open the webcam and start the real-time detection loop.
        Presses Q or ESC to quit, S to save a screenshot.
        """
        print(f"\n[INFO] Opening camera {self.camera_index} …")
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"\n[ERROR] Cannot open camera {self.camera_index}.")
            print("  Fix 1 → Try a different index:  python emotion_detector.py --camera 1")
            print("  Fix 2 → Windows: Settings → Privacy → Camera → Allow access")
            print("  Fix 3 → macOS:   System Preferences → Security & Privacy → Camera")
            sys.exit(1)

        # Request HD resolution — camera will use closest supported size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS,           30)

        print("[INFO] Camera opened ✅")
        print("[INFO] Controls:  Q / ESC = Quit   |   S = Screenshot   |   B = Toggle bars")
        print()

        frame_idx    = 0          # total frames read
        fps          = 0.0        # rolling FPS estimate
        prev_time    = time.time()
        start_time   = time.time()
        top_emotion  = "neutral"

        # Cache last detection results so we can display them on skipped frames
        last_results = []         # list of {box, emotions} dicts

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Lost camera feed.")
                break

            frame_idx += 1

            # ── FPS calculation (exponential moving average) ───────────────
            now       = time.time()
            fps       = 0.92 * fps + 0.08 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            elapsed   = now - start_time

            # ── Run FER detection every skip_frames frames ─────────────────
            if frame_idx % self.skip_frames == 0:
                if self.fer is not None:
                    # Downscale to 50% for faster inference, then scale boxes back
                    small, ratio = self._downscale(frame, scale=0.5)
                    try:
                        raw = self.fer.detect_emotions(small)
                        # Scale bounding boxes back to original frame size
                        last_results = []
                        for r in raw:
                            bx, by, bw, bh = r["box"]
                            last_results.append({
                                "box":      (int(bx * ratio), int(by * ratio),
                                             int(bw * ratio), int(bh * ratio)),
                                "emotions": r["emotions"]
                            })
                    except Exception:
                        pass   # Occasional FER errors on blurry/partial frames
                else:
                    # Haar fallback
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small_gray, ratio = self._downscale(
                        gray, scale=0.5)
                    faces = self._detect_haar(small_gray)
                    last_results = []
                    for (fx, fy, fw, fh) in faces:
                        last_results.append({
                            "box":      (int(fx * ratio), int(fy * ratio),
                                         int(fw * ratio), int(fh * ratio)),
                            "emotions": None
                        })

            # ── Annotate cached results on every frame ─────────────────────
            for face_id, result in enumerate(last_results, start=1):
                x, y, w, h = result["box"]
                emotions    = result.get("emotions")
                self._annotate_face(frame, x, y, w, h, emotions, face_id)
                if emotions:
                    top_emotion = max(emotions, key=emotions.get)

            # ── HUD overlay ───────────────────────────────────────────────
            draw_hud(frame, fps, len(last_results), top_emotion, elapsed)

            # ── Watermark ─────────────────────────────────────────────────
            fh, fw = frame.shape[:2]
            cv2.putText(frame, "Ello Emotion Detector",
                        (fw - 220, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 70), 1)

            # ── Display ───────────────────────────────────────────────────
            cv2.imshow("Ello — Face Emotion Detector  [ Q = Quit | S = Screenshot ]",
                       frame)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):          # Q or ESC → quit
                print("\n[INFO] Quit key pressed. Exiting …")
                break

            elif key in (ord('s'), ord('S')):             # S → screenshot
                fname = os.path.join(
                    SCREENSHOT_DIR,
                    f"emotion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(fname, frame)
                print(f"[INFO] Screenshot saved → {fname}")

            elif key in (ord('b'), ord('B')):             # B → toggle bars
                self.show_bars = not self.show_bars
                state = "ON" if self.show_bars else "OFF"
                print(f"[INFO] Emotion bars: {state}")

        # ── Cleanup ────────────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Session complete — {frame_idx} frames processed.")

    # ── Single image mode ─────────────────────────────────────────────────────

    def analyse_image(self, image_path: str):
        """
        Run detection on a single image file instead of live camera.
        Saves annotated result as  result_<original_filename>.jpg
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            sys.exit(1)

        frame = cv2.imread(image_path)
        if frame is None:
            print("[ERROR] Could not decode image file.")
            sys.exit(1)

        print(f"[INFO] Analysing image: {image_path}")

        if self.fer is not None:
            results = self.fer.detect_emotions(frame)
            print(f"[INFO] Found {len(results)} face(s)")
            for i, r in enumerate(results, 1):
                x, y, w, h = r["box"]
                emos = r["emotions"]
                self._annotate_face(frame, x, y, w, h, emos, face_id=i)
                top = max(emos, key=emos.get)
                print(f"\n  Face {i}: {EMOTION_LABELS.get(top, top)} "
                      f"({emos[top]*100:.0f}%)")
                for emo, sc in sorted(emos.items(), key=lambda e: e[1], reverse=True):
                    bar = "█" * int(sc * 20)
                    print(f"    {emo:<10} {bar:<20} {sc*100:5.1f}%")
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detect_haar(gray)
            print(f"[INFO] Found {len(faces)} face(s) (no emotion — FER not installed)")
            for (x, y, w, h) in faces:
                self._annotate_face(frame, x, y, w, h)

        out = "result_" + os.path.basename(image_path)
        cv2.imwrite(out, frame)
        print(f"\n[INFO] Result saved → {out}")
        cv2.imshow("Result — press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Ello Real-Time Face & Emotion Recognition",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (0=built-in  1=USB  etc.)")
    p.add_argument("--skip",   type=int, default=2,
                   help="Run detection every N frames (2=faster, 1=slower+accurate)")
    p.add_argument("--mtcnn",  action="store_true",
                   help="Use MTCNN face detector (more accurate, slower)")
    p.add_argument("--nobars", action="store_true",
                   help="Hide emotion probability bar chart")
    p.add_argument("--image",  type=str, default=None,
                   help="Analyse a single image instead of camera")
    return p.parse_args()


def main():
    args     = parse_args()
    analyzer = FaceAnalyzer(
        camera_index = args.camera,
        skip_frames  = max(1, args.skip),
        use_mtcnn    = args.mtcnn,
        show_bars    = not args.nobars,
    )

    if args.image:
        analyzer.analyse_image(args.image)
    else:
        analyzer.run()


if __name__ == "__main__":
    main()
