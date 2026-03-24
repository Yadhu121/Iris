import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import sounddevice as sd
import queue
import threading
from faster_whisper import WhisperModel
import tkinter as tk
from deep_translator import GoogleTranslator
import subprocess
import os, sys, math, pickle, json

_DEFAULT_APP_MAP = {
    'A': r'C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE',
    'C': r'C:\Windows\System32\calc.exe',
    'D': r'C:\Windows\explorer.exe',
    'E': r'C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE',
    'F': r'C:\Windows\System32\notepad.exe',
    'G': r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    'I': r'C:\Windows\System32\notepad.exe',
    'K': r'C:\Windows\System32\notepad.exe',
    'L': r'C:\Windows\explorer.exe',
    'P': r'C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE',
    'T': r'C:\Windows\System32\cmd.exe',
    'U': r'C:\Windows\System32\notepad.exe',
    'V': r'C:\Program Files\Microsoft VS Code\Code.exe',
    'X': r'C:\Windows\System32\taskmgr.exe',
    'Z': r'C:\Windows\System32\notepad.exe',
}

def _load_app_map():
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    mappings_file = os.path.join(script_dir, "mappings.json")
    if os.path.exists(mappings_file):
        try:
            with open(mappings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = {k.upper(): v for k, v in data.items() if v and v.strip()}
            if loaded:
                print(f"[AppMap] Loaded {len(loaded)} mappings from mappings.json")
                return loaded
        except Exception as e:
            print(f"[AppMap] Could not read mappings.json ({e}), using defaults")
    return {k: v for k, v in _DEFAULT_APP_MAP.items() if v}

APP_MAP   = _load_app_map()

def _app_names_from_map(amap):
    names = {}
    for letter, path in amap.items():
        try:    names[letter] = os.path.splitext(os.path.basename(path))[0]
        except: names[letter] = path
    return names

APP_NAMES = _app_names_from_map(APP_MAP)

LETTERS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
MODEL_CACHE = "air_letter_model.pkl"
_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
]

def _render_letter(letter, font, scale, thickness, angle, tx, ty):
    img = np.zeros((96, 96), np.uint8)
    (tw, th), _ = cv2.getTextSize(letter, font, scale, thickness)
    ox = (96 - tw) // 2 + tx
    oy = (96 + th) // 2 + ty
    cv2.putText(img, letter, (max(0,ox), max(th,oy)), font, scale, 255, thickness, cv2.LINE_AA)
    M  = cv2.getRotationMatrix2D((48,48), angle, 1.0)
    img = cv2.warpAffine(img, M, (96,96), borderMode=cv2.BORDER_CONSTANT)
    return cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

def _elastic_distort(img, alpha=6, sigma=2):
    rng = np.random.default_rng()
    dx  = cv2.GaussianBlur((rng.random((28,28))*2-1).astype(np.float32),(5,5),sigma)*alpha
    dy  = cv2.GaussianBlur((rng.random((28,28))*2-1).astype(np.float32),(5,5),sigma)*alpha
    x,y = np.meshgrid(np.arange(28),np.arange(28))
    return cv2.remap(img, np.clip(x+dx,0,27).astype(np.float32),
                          np.clip(y+dy,0,27).astype(np.float32), cv2.INTER_LINEAR)

def _add_noise(img, amount=15):
    noise = np.random.randint(-amount,amount,img.shape,np.int16)
    return np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)

def _img_to_hog(img):
    from skimage.feature import hog
    return hog(img,orientations=9,pixels_per_cell=(7,7),
               cells_per_block=(2,2),visualize=False).astype(np.float32)

def _generate_training_data():
    print("[Writing] Generating synthetic training data (first run only)...")
    X, y = [], []
    rng         = np.random.default_rng(42)
    rotations   = list(range(-20,21,5))
    scales      = [1.0,1.3,1.6,1.9,2.2]
    thicknesses = [1,2,3,4]
    offsets     = [(0,0),(-8,0),(8,0),(0,-8),(0,8),(-5,-5),(5,5),(-5,5),(5,-5)]
    for idx, letter in enumerate(LETTERS):
        samples = []
        for font in _FONTS:
            for scale in scales:
                for thick in thicknesses:
                    for angle in rotations:
                        for tx,ty in offsets:
                            img = _render_letter(letter,font,scale,thick,angle,tx,ty)
                            if np.count_nonzero(img) < 10: continue
                            samples.append(img)
                            samples.append(_elastic_distort(img))
                            samples.append(_add_noise(img))
        rng.shuffle(samples); samples = samples[:400]
        for img in samples:
            X.append(_img_to_hog(img)); y.append(idx)
        if (idx+1)%5==0:
            print(f"  {idx+1}/26 letters done...")
    return np.array(X,np.float32), np.array(y,np.int32)

def _canvas_to_hog(canvas):
    from skimage.feature import hog
    gray   = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None: return None
    x,y,w,h = cv2.boundingRect(coords)
    pad  = max(w,h)//4
    x=max(0,x-pad); y=max(0,y-pad)
    w=min(canvas.shape[1]-x,w+2*pad); h=min(canvas.shape[0]-y,h+2*pad)
    crop = gray[y:y+h,x:x+w]
    size = max(w,h)
    sq   = np.zeros((size,size),np.uint8)
    sq[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w] = crop
    img28 = cv2.resize(sq,(28,28),interpolation=cv2.INTER_AREA)
    img28 = cv2.equalizeHist(img28)
    return hog(img28,orientations=9,pixels_per_cell=(7,7),
               cells_per_block=(2,2),visualize=False).astype(np.float32)

class LetterRecognizer:
    def __init__(self):
        self.pipe = None
        if os.path.exists(MODEL_CACHE):
            print("[Writing] Loading cached model...")
            try:
                with open(MODEL_CACHE,'rb') as f: self.pipe = pickle.load(f)
                print("[Writing] Model ready."); return
            except Exception as e:
                print(f"[Writing] Cache broken ({e}), retraining...")
        self._train()

    def _train(self):
        try:    from skimage.feature import hog
        except ImportError: sys.exit("[ERROR] Run: pip install scikit-image")
        try:    from sklearn.ensemble import RandomForestClassifier
        except ImportError: sys.exit("[ERROR] Run: pip install scikit-learn")
        X, y = _generate_training_data()
        print(f"[Writing] Training on {len(X)} samples...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline  import Pipeline
        from sklearn.preprocessing import StandardScaler
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(n_estimators=200,max_depth=20,
                                              min_samples_leaf=2,n_jobs=-1,random_state=42)),
        ])
        self.pipe.fit(X, y)
        print("[Writing] Training done.")
        with open(MODEL_CACHE,'wb') as f: pickle.dump(self.pipe,f)

    def predict(self, canvas):
        if self.pipe is None: return None, 0.0
        feat = _canvas_to_hog(canvas)
        if feat is None: return None, 0.0
        proba = self.pipe.predict_proba(feat.reshape(1,-1))[0]
        idx   = int(np.argmax(proba))
        return LETTERS[idx], float(proba[idx])

def launch_app(letter):
    path = APP_MAP.get(letter); name = APP_NAMES.get(letter,'?')
    if not path: print(f"[Writing] No mapping for '{letter}'"); return False
    try:    subprocess.Popen(path,shell=True); print(f"[Writing] Launching {name}"); return True
    except Exception as e: print(f"[Writing] Launch error: {e}"); return False

class TranslateEngine:
    def translate(self, text, mode):
        try:
            if mode == "ml-en":
                return GoogleTranslator(source='ml', target='en').translate(text)
            else:
                return GoogleTranslator(source='en', target='ml').translate(text)
        except Exception as e:
            print(f"TRANSLATION ERROR: {e}"); return text

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.7, min_tracking_confidence=0.5
)

screen_width, screen_height = pyautogui.size()
cam_width, cam_height       = 640, 480
frame_reduction             = 100

MOUSE   = "Mouse"
VOICE   = "Voice"
WRITING = "Writing"

current_mode    = MOUSE
pending_mode    = None
mode_hold_start = None
MODE_HOLD_SECONDS = 1.0

translate_mode = None
gesture_active = False
is_Fist        = False

smoothing = 5
prev_x, prev_y = 0, 0
is_Left_Click = is_Right_Click = is_Enter_Click = False
is_Dragging = is_altTab = is_Backspace = False
is_arrow_left = is_arrow_right = False
last_tab_time  = 0
CLICK_THRESHOLD = 0.05

is_Recording = False

HAND_COLORS = {"Right": (0, 210, 100), "Left": (200, 80, 0)}

writing_enabled  = False
writing_fist_cd  = 0.0
writing_prev_pt  = None
writing_canvas   = None
writing_detected = None
writing_det_t    = 0.0

OV_W, OV_H     = 200, 180
SKEL_W, SKEL_H = 200, 150

C_GREEN   = "#4ade80"
C_ORANGE  = "#fb923c"
C_RED     = "#f87171"

SKEL_COLORS = {"Right": "#00ff78", "Left": "#ff6400"}

overlay = tk.Tk()
overlay.overrideredirect(True)
overlay.attributes('-topmost', True)
overlay.configure(bg="black")
overlay.resizable(False, False)
overlay.geometry(f"{OV_W}x{OV_H}+{screen_width-OV_W-12}+{screen_height-OV_H-12}")

skel_canvas = tk.Canvas(overlay, width=SKEL_W, height=SKEL_H,
                          bg="black", highlightthickness=0)
skel_canvas.pack()

writing_info_lbl = tk.Label(skel_canvas, text="", fg="white", bg="black",
                              font=("Segoe UI", 9))
writing_info_lbl.place(relx=0.5, rely=0.88, anchor="center")

gesture_label = tk.Label(overlay, text="Ready", fg="#888888", bg="black",
                           font=("Segoe UI", 8), anchor="w")
gesture_label.pack(fill="x", padx=6)

_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

def _draw_skeleton_on_overlay(hands_data):
    skel_canvas.delete("all")
    for label, lm in hands_data:
        color = SKEL_COLORS.get(label, "#ffffff")
        pts   = [(int(l.x * SKEL_W), int(l.y * SKEL_H)) for l in lm]
        for s, e in _CONNECTIONS:
            if s < len(pts) and e < len(pts):
                skel_canvas.create_line(pts[s][0], pts[s][1],
                                         pts[e][0], pts[e][1],
                                         fill=color, width=2, capstyle="round")
        for idx, (px, py) in enumerate(pts):
            r = 4 if idx in (4,8,12,16,20) else 2
            skel_canvas.create_oval(px-r, py-r, px+r, py+r, fill=color, outline="")
        if pts:
            skel_canvas.create_text(pts[0][0], max(pts[0][1]-10, 8),
                                     text=label[0], fill=color,
                                     font=("Segoe UI", 7, "bold"))
    writing_info_lbl.lift()

def _set_gesture_label(text, color=None):
    gesture_label.config(text=text, fg=color or "#888888")

def _set_writing_info(text):
    writing_info_lbl.config(text=text)

class VoiceRecorder:
    def __init__(self, model_size="small"):
        self.model        = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=4)
        self.audio_queue  = queue.Queue()
        self.audio_buffer = []
        self.is_recording = False
        self.stream = sd.InputStream(samplerate=16000, channels=1,
                                     callback=self._audio_callback, blocksize=1024)
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording: self.audio_queue.put(indata.copy())

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True; self.audio_buffer = []
            print("Recording audio"); return True
        return False

    def stop_recording(self, mode=None):
        if self.is_recording:
            self.is_recording = False
            while not self.audio_queue.empty():
                self.audio_buffer.append(self.audio_queue.get())
            threading.Thread(target=self._transcribe,
                             args=(self.audio_buffer.copy(), mode), daemon=True).start()
            return True
        return False

    def update(self):
        if self.is_recording:
            while not self.audio_queue.empty(): self.audio_buffer.append(self.audio_queue.get())
        else:
            while not self.audio_queue.empty(): self.audio_queue.get()

    def _transcribe(self, buffer, mode=None):
        if not buffer: return
        try:
            audio_np = np.concatenate(buffer).flatten()
            if len(audio_np) < 16000*0.3: return
            lang = "ml" if mode=="ml-en" else "en"
            segments, _ = self.model.transcribe(
                audio_np, beam_size=5, language=lang, vad_filter=True,
                vad_parameters={"threshold":0.5,"min_speech_duration_ms":250,
                                 "min_silence_duration_ms":500},
                condition_on_previous_text=False)
            transcription = " ".join(s.text.strip() for s in segments)
            if transcription:
                print(f"TRANSCRIBED: {transcription}")
                if mode:
                    translated = translator.translate(transcription, mode)
                    print(f"TRANSLATED: {translated}")
                    pyautogui.write(translated, interval=0.01)
                else:
                    pyautogui.write(transcription, interval=0.01)
        except Exception as e:
            print(f"ERROR: {e}")

    def cleanup(self):
        self.stream.stop(); self.stream.close()

voice      = VoiceRecorder(model_size="small")
translator = TranslateEngine()
recognizer = LetterRecognizer()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

ret, _frame = cap.read()
_fh, _fw    = _frame.shape[:2] if ret else (cam_height, cam_width)
writing_canvas = np.zeros((_fh, _fw, 3), np.uint8)

def count_fingers(landmarks, hand_label):
    fingers = []
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return fingers

def calculate_distance(p1, p2):
    return np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)

def isLeftTouching(h):      return calculate_distance(h.landmark[4],h.landmark[8])  < CLICK_THRESHOLD
def isRightTouching(h):     return calculate_distance(h.landmark[4],h.landmark[12]) < CLICK_THRESHOLD
def isEnterTouching(h):     return calculate_distance(h.landmark[4],h.landmark[16]) < CLICK_THRESHOLD
def isBackspaceTouching(h): return calculate_distance(h.landmark[4],h.landmark[20]) < CLICK_THRESHOLD

def move_cursor(hand_landmarks, frame_width, frame_height):
    global prev_x, prev_y
    cp = hand_landmarks.landmark[9]
    x  = int(cp.x*frame_width);  y = int(cp.y*frame_height)
    sx = np.interp(x,[frame_reduction,frame_width-frame_reduction],[0,screen_width])
    sy = np.interp(y,[frame_reduction,frame_height-frame_reduction],[0,screen_height])
    nx = prev_x+(sx-prev_x)/smoothing
    ny = prev_y+(sy-prev_y)/smoothing
    pyautogui.moveTo(nx, ny)
    prev_x, prev_y = nx, ny

def get_hand_label(index, results):
    return results.multi_handedness[index].classification[0].label

def draw_skeleton_cv2(frame, hand_landmarks, hand_label, frame_width, frame_height):
    color = HAND_COLORS.get(hand_label, (255,255,255))
    pts   = [(int(lm.x*frame_width), int(lm.y*frame_height)) for lm in hand_landmarks.landmark]
    for s, e in mp_hands.HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], color, 2)
    for idx, p in enumerate(pts):
        r = 6 if idx in (4,8,12,16,20) else 3
        cv2.circle(frame, p, r, color, -1)

def is_fist_writing(lm):
    tips=[8,12,16,20]; pips=[6,10,14,18]
    return all(lm.landmark[t].y > lm.landmark[p].y for t,p in zip(tips,pips))

def is_thumb_pointing_left(lm):  return lm[4].x < lm[2].x and lm[4].y > lm[3].y
def is_thumb_pointing_right(lm): return lm[4].x > lm[2].x and lm[4].y > lm[3].y

def detect_mode_gesture(fingers):
    if fingers[1:] == [0,1,1,1]: return MOUSE
    if fingers[1:] == [1,1,1,0]: return VOICE
    if fingers[1:] == [0,0,1,1]: return WRITING
    return None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    voice.update()
    frame        = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results      = hands.process(rgb_frame)

    overlay_hands = []

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        if num_hands >= 2:
            modes_detected = []
            for j in range(num_hands):
                lm_j    = results.multi_hand_landmarks[j].landmark
                label_j = get_hand_label(j, results)
                f_j     = count_fingers(lm_j, label_j)
                modes_detected.append(detect_mode_gesture(f_j))

            if (modes_detected[0] is not None and
                    modes_detected[0] == modes_detected[1]):
                if pending_mode != modes_detected[0]:
                    pending_mode    = modes_detected[0]
                    mode_hold_start = time.time()
                elif time.time()-mode_hold_start >= MODE_HOLD_SECONDS:
                    if current_mode != pending_mode:
                        current_mode = pending_mode
                        print(f"Mode → {current_mode}")
                        _set_gesture_label(f"{current_mode} mode", C_GREEN)
                        if current_mode != WRITING:
                            writing_enabled   = False
                            writing_canvas[:] = 0
                            writing_prev_pt   = None
                            _set_writing_info("")
                    pending_mode    = None
                    mode_hold_start = None
            else:
                pending_mode    = None
                mode_hold_start = None

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = get_hand_label(i, results)
            draw_skeleton_cv2(frame, hand_landmarks, hand_label, frame_width, frame_height)
            lm      = hand_landmarks.landmark
            fingers = count_fingers(lm, hand_label)

            overlay_hands.append((hand_label, lm))

            if hand_label == "Right":
                if fingers == [0,0,0,0,0]:
                    if not is_Fist:
                        is_Fist        = True
                        gesture_active = not gesture_active
                        print(f"Gestures {'ON' if gesture_active else 'OFF'}")
                        _set_gesture_label(
                            "active" if gesture_active else "inactive",
                            C_GREEN if gesture_active else "#888888"
                        )
                else:
                    is_Fist = False

            if current_mode == MOUSE and gesture_active:
                if hand_label == "Right":
                    move_cursor(hand_landmarks, frame_width, frame_height)

                    lc = isLeftTouching(hand_landmarks)
                    rc = isRightTouching(hand_landmarks)
                    ec = isEnterTouching(hand_landmarks)

                    if lc and not is_Left_Click:
                        is_Left_Click=True; pyautogui.click()
                        _set_gesture_label("Left click", C_GREEN)
                    elif not lc: is_Left_Click=False

                    if rc and not is_Right_Click:
                        is_Right_Click=True; pyautogui.rightClick()
                        _set_gesture_label("Right click", C_GREEN)
                    elif not rc: is_Right_Click=False

                    if ec and not is_Enter_Click:
                        is_Enter_Click=True; pyautogui.hotkey("enter")
                        _set_gesture_label("Enter", C_GREEN)
                    elif not ec: is_Enter_Click=False

                    if is_thumb_pointing_left(lm) and not is_arrow_left:
                        is_arrow_left=True; pyautogui.press('left')
                        _set_gesture_label("← Arrow", "#888888")
                    elif not is_thumb_pointing_left(lm): is_arrow_left=False

                elif hand_label == "Left":
                    if fingers==[0,1,1,0,0]:
                        pyautogui.scroll(5); _set_gesture_label("Scroll ↑", "#888888")
                    if fingers==[0,1,0,0,0]:
                        pyautogui.scroll(-5); _set_gesture_label("Scroll ↓", "#888888")

                    drag = isLeftTouching(hand_landmarks)
                    if drag and not is_Dragging:
                        is_Dragging=True; pyautogui.mouseDown()
                        _set_gesture_label("Dragging…", C_ORANGE)
                    if not drag and is_Dragging:
                        is_Dragging=False; pyautogui.mouseUp()
                        _set_gesture_label("Drop", "#888888")

                    if fingers==[0,0,0,0,0]:
                        if not is_altTab:
                            is_altTab=True; pyautogui.keyDown('alt')
                            pyautogui.press('tab'); last_tab_time=time.time()
                            _set_gesture_label("Alt+Tab", C_ORANGE)
                        elif time.time()-last_tab_time>0.9:
                            pyautogui.press('tab'); last_tab_time=time.time()
                    if fingers!=[0,0,0,0,0] and is_altTab:
                        is_altTab=False; pyautogui.keyUp('alt')

                    back = isBackspaceTouching(hand_landmarks)
                    if back and not is_Backspace:
                        is_Backspace=True; pyautogui.hotkey('backspace')
                        _set_gesture_label("Backspace", C_RED)
                    elif not back: is_Backspace=False

                    if is_thumb_pointing_right(lm) and not is_arrow_right:
                        is_arrow_right=True; pyautogui.press('right')
                        _set_gesture_label("→ Arrow", "#888888")
                    elif not is_thumb_pointing_right(lm): is_arrow_right=False

            elif current_mode == VOICE and gesture_active:
                if hand_label == "Right":
                    if fingers==[0,1,1,1,0] and not is_Recording:
                        is_Recording=True; voice.start_recording()
                        _set_gesture_label("● Recording…", C_RED)
                    if fingers!=[0,1,1,1,0] and is_Recording:
                        voice.stop_recording(); is_Recording=False
                        _set_gesture_label("Transcribing…", C_ORANGE)

                elif hand_label == "Left":
                    if fingers==[0,1,1,0,0] and not is_Recording:
                        is_Recording=True; translate_mode="ml-en"; voice.start_recording()
                        _set_gesture_label("● ML→EN…", C_RED)
                    elif fingers==[0,0,1,1,1] and not is_Recording:
                        is_Recording=True; translate_mode="en-ml"; voice.start_recording()
                        _set_gesture_label("● EN→ML…", C_RED)
                    if fingers not in [[0,1,1,0,0],[0,0,1,1,1]] and is_Recording:
                        voice.stop_recording(mode=translate_mode)
                        is_Recording=False; translate_mode=None
                        _set_gesture_label("Translating…", C_ORANGE)

            elif current_mode == WRITING and gesture_active:

                if hand_label == "Left":
                    fist_now = is_fist_writing(hand_landmarks)
                    if fist_now and time.time() > writing_fist_cd:
                        if not writing_enabled:
                            writing_enabled   = True
                            writing_canvas[:] = 0
                            writing_prev_pt   = None
                            writing_detected  = None
                            _set_gesture_label("Draw now", C_GREEN)
                            _set_writing_info("fist again → confirm")
                            print("[Writing] STARTED")
                        else:
                            writing_enabled = False
                            letter, conf    = recognizer.predict(writing_canvas)
                            if letter:
                                launch_app(letter)
                                writing_detected = (letter, conf)
                                writing_det_t    = time.time()
                                app_name = APP_NAMES.get(letter, '?')
                                _set_gesture_label(f"'{letter}' → {app_name}", C_GREEN)
                                _set_writing_info(f"{letter}   {conf:.0%}")
                            else:
                                _set_gesture_label("Not recognised", C_RED)
                                _set_writing_info("")
                            writing_canvas[:] = 0
                            writing_prev_pt   = None
                            print("[Writing] STOPPED")
                        writing_fist_cd = time.time() + 1.0

                if hand_label == "Right" and writing_enabled:
                    ix = int(lm[8].x*frame_width)
                    iy = int(lm[8].y*frame_height)
                    cv2.circle(frame, (ix,iy), 5, (0,255,140), -1)
                    cur_pt = (ix,iy)
                    if writing_prev_pt is not None:
                        dist = math.hypot(ix-writing_prev_pt[0], iy-writing_prev_pt[1])
                        if dist > 2:
                            cv2.line(writing_canvas, writing_prev_pt, cur_pt, (255,255,255), 18)
                            cv2.line(writing_canvas, writing_prev_pt, cur_pt, (180,220,255), 8)
                    writing_prev_pt = cur_pt
                elif hand_label == "Right" and not writing_enabled:
                    writing_prev_pt = None

    else:
        _draw_skeleton_on_overlay([])

    if results.multi_hand_landmarks:
        _draw_skeleton_on_overlay(overlay_hands)

    if writing_detected and time.time()-writing_det_t >= 4.0:
        writing_detected = None
        _set_writing_info("")

    cv2.imshow('Iris', frame)
    overlay.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

voice.cleanup()
cap.release()
cv2.destroyAllWindows()
try: overlay.destroy()
except: pass
hands.close()