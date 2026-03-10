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
from enum import Enum
from deep_translator import GoogleTranslator
import subprocess

class TranslateEngine:
    def translate(self, text, mode):
        try:
            if mode == "ml-en":
                return GoogleTranslator(source='ml', target='en').translate(text)
            else:
                return GoogleTranslator(source='en', target='ml').translate(text)
        except Exception as e:
            print(f"TRANSLATION ERROR: {e}")
            return text

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

screen_width, screen_height = pyautogui.size()
cam_width, cam_height = 640, 480
frame_reduction = 100

class Mode(Enum):
    MOUSE = "Mouse"
    KEYBOARD = "Keyboard"
    VOICE = "Voice"

current_mode = Mode.MOUSE
pending_mode = None
mode_hold_start = None
MODE_HOLD_SECONDS = 1.0

translate_mode = None

osk = None

gesture_active = False
is_Fist = False

# Mouse
smoothing = 5
prev_x, prev_y = 0, 0

is_Left_Click = False
is_Right_Click = False
is_Enter_Click = False
is_Dragging = False
is_altTab = False
is_Backspace = False
is_arrow_left = False
is_arrow_right = False

last_scroll_time = 0
last_tab_time = 0
CLICK_THRESHOLD = 0.05
# ----

# Voice
is_Recording = False
# ----


HAND_COLORS = {
    "Right": (0, 255, 120),
    "Left": (255, 100, 0)
}

# Helper functions

def count_fingers(landmarks, hand_label):
    fingers = []

    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return fingers


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def isLeftTouching(hand_landmark):
    return calculate_distance(hand_landmark.landmark[4], hand_landmark.landmark[8]) < CLICK_THRESHOLD


def isRightTouching(hand_landmark):
    return calculate_distance(hand_landmark.landmark[4], hand_landmark.landmark[12]) < CLICK_THRESHOLD


def isEnterTouching(hand_landmark):
    return calculate_distance(hand_landmark.landmark[4], hand_landmark.landmark[16]) < CLICK_THRESHOLD


def isBackspaceTouching(hand_landmark):
    return calculate_distance(hand_landmark.landmark[4], hand_landmark.landmark[20]) < CLICK_THRESHOLD


def move_cursor(hand_landmarks, frame_width, frame_height):
    global prev_x, prev_y
    control_point = hand_landmarks.landmark[9]
    x = int(control_point.x * frame_width)
    y = int(control_point.y * frame_height)
    screen_x = np.interp(x, [frame_reduction, frame_width - frame_reduction], [0, screen_width])
    screen_y = np.interp(y, [frame_reduction, frame_height - frame_reduction], [0, screen_height])
    smooth_x = prev_x + (screen_x - prev_x) / smoothing
    smooth_y = prev_y + (screen_y - prev_y) / smoothing
    pyautogui.moveTo(smooth_x, smooth_y)
    prev_x, prev_y = smooth_x, smooth_y


def get_hand_label(index, results):
    return results.multi_handedness[index].classification[0].label


def draw_skeleton(frame, hand_landmarks, hand_label, frame_width, frame_height):
    color = HAND_COLORS.get(hand_label, (255, 255, 255))
    landmark_points = []
    for lm in hand_landmarks.landmark:
        cx = int(lm.x * frame_width)
        cy = int(lm.y * frame_height)
        landmark_points.append((cx, cy))
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(frame, landmark_points[start_idx], landmark_points[end_idx], color, 2)
    for idx, point in enumerate(landmark_points):
        r = 8 if idx in (4, 8) else 4
        cv2.circle(frame, point, r, color, -1)
    wrist = landmark_points[0]
    cv2.putText(frame, hand_label, (wrist[0] - 30, wrist[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def is_thumb_pointing_left(landmarks):
    return landmarks[4].x < landmarks[2].x and landmarks[4].y > landmarks[3].y


def is_thumb_pointing_right(landmarks):
    return landmarks[4].x > landmarks[2].x and landmarks[4].y > landmarks[3].y


def detect_mode_gesture(fingers):
    if fingers[1:] == [0, 1, 1, 1]:
        return Mode.MOUSE
    elif fingers[1:] == [1, 1, 0, 0]:
        return Mode.KEYBOARD
    elif fingers[1:] == [1, 1, 1, 0]:
        return Mode.VOICE
    return None


overlay = tk.Tk()
overlay.overrideredirect(True)
overlay.attributes('-topmost', True)
overlay.attributes('-transparentcolor', 'black')
overlay.configure(bg='black')

label_width, label_height = 300, 50
overlay.geometry(f"{label_width}x{label_height}+{screen_width - label_width - 10}+10")

gesture_label = tk.Label(overlay, text="[Mouse] Gesture: None",
                          fg='cyan', bg='black',
                          font=('Arial', 14, 'bold'))
gesture_label.pack()


class VoiceRecorder:
    def __init__(self, model_size="small"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=4
        )

        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.is_recording = False
        self.recording_start_time = None

        self.stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_buffer = []
            self.recording_start_time = time.time()
            print("Recording audio")
            return True
        return False

    def stop_recording(self, mode=None):
        if self.is_recording:
            self.is_recording = False
            while not self.audio_queue.empty():
                self.audio_buffer.append(self.audio_queue.get())
            buffer_copy = self.audio_buffer.copy()
            threading.Thread(target=self._transcribe, args=(buffer_copy, mode), daemon=True).start()
            return True
        return False

    def update(self):
        if self.is_recording:
            while not self.audio_queue.empty():
                self.audio_buffer.append(self.audio_queue.get())
        else:
            while not self.audio_queue.empty():
                self.audio_queue.get()

    def _transcribe(self, buffer, mode=None):
        if len(buffer) == 0:
            return
        try:
            audio_np = np.concatenate(buffer).flatten()
            if len(audio_np) < 16000 * 0.3:
                return
            
            lang = "ml" if mode == "ml-en" else "en"

            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language=lang,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500
                },
                condition_on_previous_text=False
            )
            transcription = " ".join(segment.text.strip() for segment in segments)
            
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
        self.stream.stop()
        self.stream.close()


voice = VoiceRecorder(model_size="small")
translator = TranslateEngine()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    voice.update()

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        if num_hands >= 2:
            modes_detected = []
            for j in range(num_hands):
                lm_j = results.multi_hand_landmarks[j].landmark
                label_j = get_hand_label(j, results)
                f_j = count_fingers(lm_j, label_j)
                modes_detected.append(detect_mode_gesture(f_j))

            if modes_detected[0] is not None and modes_detected[0] == modes_detected[1]:
                if pending_mode != modes_detected[0]:
                    pending_mode = modes_detected[0]
                    mode_hold_start = time.time()
                elif time.time() - mode_hold_start >= MODE_HOLD_SECONDS:
                    if current_mode != pending_mode:
                        current_mode = pending_mode
                        print(f"Mode switched to: {current_mode.value}")
                        if current_mode == Mode.KEYBOARD:
                            if osk is None:
                                osk = subprocess.Popen("osk.exe", shell=True)
                        else:
                            if osk is not None:
                                osk.terminate()
                                osk = None
                        gesture_label.config(text=f"[{current_mode.value}] Mode Active")
                        overlay.update()
                    pending_mode = None
                    mode_hold_start = None
            else:
                pending_mode = None
                mode_hold_start = None

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = get_hand_label(i, results)
            draw_skeleton(frame, hand_landmarks, hand_label, frame_width, frame_height)

            lm = hand_landmarks.landmark
            fingers = count_fingers(lm, hand_label)

            if hand_label == "Right":
                if fingers == [0, 0, 0, 0, 0]:
                    if not is_Fist:
                        is_Fist = True
                        gesture_active = not gesture_active
                        print(f"Gestures {'ACTIVATED' if gesture_active else 'DEACTIVATED'}")
                        gesture_label.config(text=f"[{current_mode.value}] {'ON' if gesture_active else 'OFF'}")
                        overlay.update()
                else:
                    is_Fist = False

            if current_mode == Mode.MOUSE and gesture_active:

                if hand_label == "Right":
                    move_cursor(hand_landmarks, frame_width, frame_height)

                    is_leftClick = isLeftTouching(hand_landmarks)
                    is_rightClick = isRightTouching(hand_landmarks)
                    is_EnterClick = isEnterTouching(hand_landmarks)

                    if is_leftClick and not is_Left_Click:
                        is_Left_Click = True
                        pyautogui.click()
                        print("Left click")
                    elif not is_leftClick:
                        is_Left_Click = False

                    if is_rightClick and not is_Right_Click:
                        is_Right_Click = True
                        pyautogui.rightClick()
                        print("Right click")
                    elif not is_rightClick:
                        is_Right_Click = False

                    if is_EnterClick and not is_Enter_Click:
                        is_Enter_Click = True
                        pyautogui.hotkey("enter")
                        print("Enter")
                    elif not is_EnterClick:
                        is_Enter_Click = False

                    if is_thumb_pointing_left(lm) and not is_arrow_left:
                        is_arrow_left = True
                        pyautogui.press('left')
                        print("Left arrow")
                    elif not is_thumb_pointing_left(lm):
                        is_arrow_left = False

                    # if is_thumb_pointing_right(lm) and not is_arrow_right:
                    #     is_arrow_right = True
                    #     pyautogui.press('right')
                    #     print("Right arrow")
                    # elif not is_thumb_pointing_right(lm):
                    #     is_arrow_right = False

                elif hand_label == "Left":
                    if fingers == [0, 1, 1, 0, 0]:
                        pyautogui.scroll(5)
                        print("Scroll up")
                        # if time.time() - last_scroll_time > 0.2:
                        #     pyautogui.scroll(50)
                        #     last_scroll_time = time.time()
                        #     print("Scroll up")

                    if fingers == [0, 1, 0, 0, 0]:
                        pyautogui.scroll(-5)
                        print("Scroll up")
                        # if time.time() - last_scroll_time > 0.2:
                        #     pyautogui.scroll(-50)
                        #     last_scroll_time = time.time()
                        #     print("Scroll down")

                    is_drag = isLeftTouching(hand_landmarks)
                    if is_drag and not is_Dragging:
                        is_Dragging = True
                        pyautogui.mouseDown()
                        print("Drag start")
                    if not is_drag and is_Dragging:
                        is_Dragging = False
                        pyautogui.mouseUp()
                        print("Drag end")

                    if fingers == [0, 0, 0, 0, 0]:
                        if not is_altTab:
                            is_altTab = True
                            pyautogui.keyDown('alt')
                            pyautogui.press('tab')
                            last_tab_time = time.time()
                            print("Alt tab")
                        else:
                            if time.time() - last_tab_time > 0.9:
                                pyautogui.press('tab')
                                last_tab_time = time.time()
                    if fingers != [0, 0, 0, 0, 0] and is_altTab:
                        is_altTab = False
                        pyautogui.keyUp('alt')

                    is_back = isBackspaceTouching(hand_landmarks)
                    if is_back and not is_Backspace:
                        is_Backspace = True
                        pyautogui.hotkey('backspace')
                        print("Backspace")
                    elif not is_back:
                        is_Backspace = False

                    if is_thumb_pointing_right(lm) and not is_arrow_right:
                        is_arrow_right = True
                        pyautogui.press('right')
                        print("Right arrow")
                    elif not is_thumb_pointing_right(lm):
                        is_arrow_right = False

            elif current_mode == Mode.KEYBOARD and gesture_active:

                index_tip = lm[8]
                tip_screen_x = int(np.interp(index_tip.x * frame_width,
                    [frame_reduction, frame_width - frame_reduction], [0, screen_width]))
                tip_screen_y = int(np.interp(index_tip.y * frame_height,
                    [frame_reduction, frame_height - frame_reduction], [0, screen_height]))

                if fingers == [1, 1, 1, 1, 1]:
                    if hand_label == "Right":
                        move_cursor(hand_landmarks, frame_width, frame_height)

                    is_leftClick = isLeftTouching(hand_landmarks)
                    # is_rightClick = isRightTouching(hand_landmarks)
                    # is_EnterClick = isEnterTouching(hand_landmarks)

                    if is_leftClick and not is_Left_Click:
                        is_Left_Click = True
                        pyautogui.click()
                        print("Left click")
                    if not is_leftClick:
                        is_Left_Click = False

            elif current_mode == Mode.VOICE and gesture_active:

                if hand_label == "Right":
                    if fingers == [0, 1, 1, 1, 0] and not is_Recording:
                        is_Recording = True
                        voice.start_recording()
                        gesture_label.config(text="[Voice] Recording...")
                        overlay.update()

                    if fingers != [0, 1, 1, 1, 0] and is_Recording:
                        voice.stop_recording()
                        is_Recording = False
                        gesture_label.config(text="[Voice] Transcribing...")
                        overlay.update()

                elif hand_label == "Left":
                    if fingers == [0, 1, 1, 0, 0] and not is_Recording:
                        is_Recording = True
                        translate_mode = "ml-en"
                        voice.start_recording()
                        gesture_label.config(text="[Voice] ML→EN Recording...")
                        overlay.update()

                    elif fingers == [0, 0, 1, 1, 1] and not is_Recording:
                        is_Recording = True
                        translate_mode = "en-ml"
                        voice.start_recording()
                        gesture_label.config(text="[Voice] EN→ML Recording...")
                        overlay.update()

                    if fingers not in [[0, 1, 1, 0, 0], [0, 0, 1, 1, 1]] and is_Recording:
                        voice.stop_recording(mode=translate_mode)
                        is_Recording = False
                        translate_mode = None
                        gesture_label.config(text="[Voice] Translating...")
                        overlay.update()

    cv2.imshow('Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

voice.cleanup()
cap.release()
cv2.destroyAllWindows()
overlay.destroy()
hands.close()
osk.terminate()