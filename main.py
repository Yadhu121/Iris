import cv2
import mediapipe as mp
import pyautogui
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
from collections import deque
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

RATE = 16000
CHUNK_SIZE = 1024
screenW, screenH = pyautogui.size()

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.smoothX, self.smoothY = 0, 0
        self.smoothing = 0.5

        self.last_left_click = False
        self.last_double_click = False
        self.last_enter = False

        self.gesture_history = deque(maxlen=3)

    def distance(self, p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

    def is_fist(self, lm):
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        ring_tip = lm[16]
        pinky_tip = lm[20]
        
        palm_base = lm[0]
        
        threshold = 0.15
        
        return (self.distance(thumb_tip, palm_base) < threshold and
                self.distance(index_tip, palm_base) < threshold and
                self.distance(middle_tip, palm_base) < threshold and
                self.distance(ring_tip, palm_base) < threshold and
                self.distance(pinky_tip, palm_base) < threshold)

    def is_index_thumb_touch(self, lm):
        return self.distance(lm[4], lm[8]) < 0.05

    def is_middle_thumb_touch(self, lm):
        return self.distance(lm[4], lm[12]) < 0.05

    def is_ring_thumb_touch(self, lm):
        return self.distance(lm[4], lm[16]) < 0.05

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        gesture_type = None

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            middleBase = lm[9]
            mouseX = screenW * middleBase.x
            mouseY = screenH * middleBase.y
            self.smoothX = self.smoothX * self.smoothing + mouseX * (1 - self.smoothing)
            self.smoothY = self.smoothY * self.smoothing + mouseY * (1 - self.smoothing)
            pyautogui.moveTo(int(self.smoothX), int(self.smoothY))

            if self.is_fist(lm):
                gesture_type = "fist"
            elif self.is_index_thumb_touch(lm):
                if not self.last_left_click:
                    pyautogui.click()
                    print("Left click")
                    self.last_left_click = True
                else:
                    self.last_left_click = True
            elif self.is_middle_thumb_touch(lm):
                if not self.last_double_click:
                    pyautogui.doubleClick()
                    print("Double clikc")
                    self.last_double_click = True
                else:
                    self.last_double_click = True
            elif self.is_ring_thumb_touch(lm):
                if not self.last_enter:
                    pyautogui.press('enter')
                    print("Enter")
                    self.last_enter = True
                else:
                    self.last_enter = True
            else:
                self.last_left_click = False
                self.last_double_click = False
                self.last_enter = False

        return gesture_type

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

def main():
    gesture = GestureController()
    voice = VoiceRecorder(model_size="small")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    was_recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gesture_type = gesture.process_frame(frame)

        if gesture_type == "fist":
            if not was_recording:
                voice.start_recording()
                was_recording = True

            if voice.recording_start_time:
                duration = time.time() - voice.recording_start_time

        else:
            if was_recording:
                voice.stop_recording()
                was_recording = False

        voice.update()

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    voice.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()