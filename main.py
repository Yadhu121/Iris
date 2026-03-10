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
from keyboard import VirtualKeyboard
from enum import Enum

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

# Keyboard
is_peace = {0: False, 1: False}
was_peace = {0: False, 1: False}
# ----

HAND_COLORS = {
    "Right": (0, 255, 120),
    "Left": (255, 100, 0)
}