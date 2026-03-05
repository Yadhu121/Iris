import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import sounddevice as sd
import queue
import threading
from faster_whisper import WhisperModel
from collections import deque

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

screen_width, screen_height = pyautogui.size()
cam_width, cam_height = 640, 480

smoothing = 5
prev_x, prev_y = 0, 0

frame_reduction = 100

gesture_active = False

LeftRight_Click_Threshold = 0.05
is_Left_Click = False
is_Right_Click = False
is_Enter_Click = False
is_Fist = False
is_Recording = False

last_scroll_time = 0
is_Dragging = False

RATE = 16000
CHUNK_SIZE = 1024

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
            samplerate=RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=CHUNK_SIZE
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time, status):
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

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False

            while not self.audio_queue.empty():
                self.audio_buffer.append(self.audio_queue.get())

            print("Recorded: ")

            buffer_copy = self.audio_buffer.copy()
            threading.Thread(
                target=self._transcribe,
                args=(buffer_copy,),
                daemon=True
            ).start()

            return True
        return False

    def update(self):
        if self.is_recording:
            while not self.audio_queue.empty():
                self.audio_buffer.append(self.audio_queue.get())
        else:
            while not self.audio_queue.empty():
                self.audio_queue.get()

    def _transcribe(self, buffer):
        if len(buffer) == 0:
            return

        try:
            audio_np = np.concatenate(buffer).flatten()

            if len(audio_np) < RATE * 0.3:
                return

            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
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
                pyautogui.write(transcription, interval=0.01)
                print("TEXT PASTED")

        except Exception as e:
            print(f"ERROR: {e}")

    def cleanup(self):
        self.stream.stop()
        self.stream.close()

voice = VoiceRecorder(model_size="small")

def count_fingers(landmarks, hand_label):
    fingers = []

    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)  # left hand

    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return fingers


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def isLeftTouching(hand_landmark):
    thumb_tip = hand_landmark.landmark[4]
    index_tip = hand_landmark.landmark[8]
    
    LeftClickdistance = calculate_distance(thumb_tip, index_tip)
    return LeftClickdistance < LeftRight_Click_Threshold

def isRightTouching(hand_landmark):
    thumb_tip = hand_landmark.landmark[4]
    middle_tip = hand_landmark.landmark[12]
    
    RightClickdistance = calculate_distance(thumb_tip, middle_tip)
    return RightClickdistance < LeftRight_Click_Threshold

def isEnterTouching(hand_landmark):
    thumb_tip = hand_landmark.landmark[4]
    ring_tip = hand_landmark.landmark[16]

    EnterDistance = calculate_distance(thumb_tip, ring_tip)
    return EnterDistance < LeftRight_Click_Threshold

def move_cursor(hand_landmarks, frame_width, frame_height):
    global prev_x, prev_y

    control_point = hand_landmarks.landmark[9]

    x = int(control_point.x * frame_width)
    y = int(control_point.y * frame_height)

    screen_x = np.interp(x, [frame_reduction, frame_width - frame_reduction], 
                         [0, screen_width])
    screen_y = np.interp(y, [frame_reduction, frame_height - frame_reduction], 
                         [0, screen_height])

    smooth_x = prev_x + (screen_x - prev_x) / smoothing
    smooth_y = prev_y + (screen_y - prev_y) / smoothing

    #smooth_x = prev_x + (screen_x - prev_x)
    #smooth_y = prev_y + (screen_y - prev_y)

    pyautogui.moveTo(smooth_x, smooth_y)

    prev_x, prev_y = smooth_x, smooth_y

    return x, y

def get_hand_label(index, results):
    handedness = results.multi_handedness[index].classification[0].label
    return handedness

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

def draw_hand_label(frame, hand_landmarks, label, frame_width, frame_height):
    wrist = hand_landmarks.landmark[0]
    cx = int(wrist.x * frame_width)
    cy = int(wrist.y * frame_height)

    color = (0, 255, 0) if label == "Right" else (255, 100, 0)
    cv2.putText(frame, label, (cx - 30, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

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
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = get_hand_label(i, results)
            draw_hand_label(frame, hand_landmarks, hand_label, frame_width, frame_height)
        
            fingers = count_fingers(hand_landmarks.landmark, hand_label)
            finger_count = sum(fingers)
        
            Right_gesture = None
            Left_gesture = None

            if hand_label == "Right":
                if fingers == [0, 0, 0, 0, 0] and not is_Fist:
                    is_Fist = True
                    Right_gesture = "FIST"
            
                if fingers != [0, 0, 0, 0, 0]:
                    is_Fist = False
            
                if gesture_active:
                    move_cursor(hand_landmarks, frame_width, frame_height)

                if fingers == [0, 1, 1, 1, 0] and not is_Recording:
                    is_Recording = True
                    Right_gesture = "Record"

                if fingers != [0, 1, 1, 1, 0] and is_Recording:
                    voice.stop_recording()
                    is_Recording = False

                is_leftClick = isLeftTouching(hand_landmarks)
                is_rightClick = isRightTouching(hand_landmarks)
                is_EnterClick = isEnterTouching(hand_landmarks)

                if is_leftClick and not is_Left_Click:
                    is_Left_Click = True
                    Right_gesture = "Left Click"
            
                if not is_leftClick:
                    is_Left_Click = False

                if is_rightClick and not is_Right_Click:
                    is_Right_Click = True
                    Right_gesture = "Right Click"
            
                if not is_rightClick:
                    is_Right_Click = False

                if is_EnterClick and not is_Enter_Click:
                    is_Enter_Click = True
                    Right_gesture = "Enter Click"

                if not is_EnterClick:
                    is_Enter_Click = False

                if Right_gesture:
                    if Right_gesture == "FIST":
                        gesture_active = not gesture_active
                        print(f"Gestures {'ACTIVATED' if gesture_active else 'DEACTIVATED'}")
                
                    if Right_gesture == "Record":
                        voice.start_recording()

                    elif Right_gesture == "Left Click" and gesture_active:
                        pyautogui.click()
                        print("Left click")

                    elif Right_gesture == "Right Click" and gesture_active:
                        pyautogui.rightClick()
                        print("Right click")

                    elif Right_gesture == "Enter Click" and gesture_active:
                        pyautogui.hotkey("enter")
                        print("Enter")

            elif hand_label == "Left":

                if fingers == [0, 1, 1, 0, 0] and gesture_active:
                    if time.time() - last_scroll_time > 0.2:
                        pyautogui.scroll(1)
                        last_scroll_time = time.time()

                if fingers == [0, 1, 0, 0, 0] and gesture_active:
                    if time.time() - last_scroll_time > 0.2:
                        pyautogui.scroll(-1)
                        last_scroll_time = time.time()

                is_drag = isLeftTouching(hand_landmarks)

                if is_drag and not is_Dragging:
                    is_Dragging = True
                    pyautogui.mouseDown()
               
                if not is_drag and is_Dragging:
                    is_Dragging = False
                    pyautogui.mouseUp()

    cv2.imshow('Gesture Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

voice.cleanup()
cap.release()
cv2.destroyAllWindows()
hands.close()