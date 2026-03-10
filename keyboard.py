import tkinter as tk
import pyautogui
import time


class VirtualKeyboard:
    KEYS = [
        ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'BKSP'],
        ['TAB', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
        ['CAPS', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'ENTER'],
        ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'SHIFT_R'],
        ['SPACE']
    ]

    KEY_W = 58
    KEY_H = 52
    PADDING = 6

    SPECIAL_WIDTHS = {
        'BKSP': 100,
        'TAB': 78,
        'CAPS': 88,
        'ENTER': 108,
        'SHIFT': 120,
        'SHIFT_R': 120,
        'SPACE': 460
    }

    KEY_MAP = {
        'BKSP': 'backspace',
        'ENTER': 'enter',
        'TAB': 'tab',
        'CAPS': 'capslock',
        'SHIFT': 'shift',
        'SHIFT_R': 'shift',
        'SPACE': 'space',
        '\\': 'backslash',
        ';': 'semicolon',
        "'": 'apostrophe',
        '`': 'grave',
        '-': 'minus',
        '=': 'equal',
        '[': 'bracketleft',
        ']': 'bracketright',
        ',': 'comma',
        '.': 'period',
        '/': 'slash',
    }

    FINGER_COLORS = ['#00d4ff', '#ff4d4d']

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.visible = False
        self.key_rects = {}
        self.hovered = {0: None, 1: None}
        self.pressed_cooldown = {}
        self.COOLDOWN = 0.6
        self.flash_pending = {}
        self.kb_screen_x = 0
        self.kb_screen_y = 0

        self.win = tk.Toplevel()
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        self.win.attributes('-alpha', 0.88)
        self.win.configure(bg='#1a1a2e')
        self.win.withdraw()

        self._build_keyboard()
        self._place_window()

        self.finger_overlay = tk.Toplevel()
        self.finger_overlay.overrideredirect(True)
        self.finger_overlay.attributes('-topmost', True)
        self.finger_overlay.attributes('-transparentcolor', 'black')
        self.finger_overlay.configure(bg='black')
        self.finger_overlay.geometry(f'{screen_w}x{screen_h}+0+0')
        self.finger_overlay.withdraw()

        self.fcanvas = tk.Canvas(
            self.finger_overlay,
            width=screen_w,
            height=screen_h,
            bg='black',
            highlightthickness=0
        )
        self.fcanvas.pack()

        self.finger_dots = {}
        for i in range(2):
            dot = self.fcanvas.create_oval(0, 0, 0, 0, fill=self.FINGER_COLORS[i], outline='white', width=2)
            label = self.fcanvas.create_text(0, 0, text='', fill=self.FINGER_COLORS[i],
                                              font=('Consolas', 9, 'bold'))
            self.finger_dots[i] = (dot, label)

    def _tag(self, key):
        return f'key_{key}'

    def _build_keyboard(self):
        self.canvas = tk.Canvas(
            self.win,
            bg='#1a1a2e',
            highlightthickness=0
        )
        self.canvas.pack()
        self._draw_keys()

    def _draw_keys(self):
        self.canvas.delete('all')
        self.key_rects.clear()

        total_width = sum(
            self.SPECIAL_WIDTHS.get(k, self.KEY_W) + self.PADDING
            for k in self.KEYS[0]
        )
        canvas_w = total_width + self.PADDING
        canvas_h = len(self.KEYS) * (self.KEY_H + self.PADDING) + self.PADDING

        self.canvas.config(width=canvas_w, height=canvas_h)
        self._canvas_w = canvas_w

        for row_i, row in enumerate(self.KEYS):
            row_width = sum(self.SPECIAL_WIDTHS.get(k, self.KEY_W) + self.PADDING for k in row)
            x = (canvas_w - row_width) // 2 + self.PADDING // 2
            y = self.PADDING + row_i * (self.KEY_H + self.PADDING)

            for key in row:
                kw = self.SPECIAL_WIDTHS.get(key, self.KEY_W)
                tag = self._tag(key)
                self.key_rects[key] = (x, y, x + kw, y + self.KEY_H)

                self.canvas.create_rectangle(
                    x, y, x + kw, y + self.KEY_H,
                    fill='#16213e',
                    outline='#0f3460',
                    width=2,
                    tags=(tag, 'rect')
                )
                self.canvas.create_text(
                    x + kw // 2, y + self.KEY_H // 2,
                    text=key.replace('_R', ''),
                    fill='#e0e0e0',
                    font=('Consolas', 11, 'bold'),
                    tags=(tag, 'label')
                )
                x += kw + self.PADDING

    def _place_window(self):
        self.win.update_idletasks()
        win_w = self.win.winfo_reqwidth()
        win_h = self.win.winfo_reqheight()
        x = (self.screen_w - win_w) // 2
        y = self.screen_h - win_h - 40
        self.win.geometry(f'+{x}+{y}')
        self.kb_screen_x = x
        self.kb_screen_y = y

    def show(self):
        self.win.deiconify()
        self.win.update_idletasks()
        self._place_window()
        self.finger_overlay.deiconify()
        self.visible = True

    def hide(self):
        self.win.withdraw()
        self.finger_overlay.withdraw()
        self.visible = False
        self.hovered = {0: None, 1: None}
        self.flash_pending.clear()

    def toggle(self):
        self.hide() if self.visible else self.show()

    def _rect_tag(self, key):
        tag = self._tag(key)
        items = self.canvas.find_withtag(tag)
        for item in items:
            if 'rect' in self.canvas.gettags(item):
                return item
        return None

    def _key_color(self, key):
        if key in self.flash_pending:
            return '#00b894'
        h0 = self.hovered.get(0)
        h1 = self.hovered.get(1)
        if key == h0 and key == h1:
            return '#7b2d8b'
        if key == h0:
            return '#0f4c75'
        if key == h1:
            return '#6a0572'
        return '#16213e'

    def _refresh_key(self, key):
        item = self._rect_tag(key)
        if item:
            self.canvas.itemconfig(item, fill=self._key_color(key))

    def update_hover(self, hand_idx, sx, sy):
        lx = sx - self.kb_screen_x
        ly = sy - self.kb_screen_y
        hovered_key = None

        for kid, (x1, y1, x2, y2) in self.key_rects.items():
            if x1 <= lx <= x2 and y1 <= ly <= y2:
                hovered_key = kid
                break

        old = self.hovered.get(hand_idx)
        self.hovered[hand_idx] = hovered_key

        if old and old != hovered_key:
            self._refresh_key(old)

        if hovered_key:
            self._refresh_key(hovered_key)

        self.canvas.update_idletasks()

        dot, lbl = self.finger_dots[hand_idx]
        r = 14
        self.fcanvas.coords(dot, sx - r, sy - r, sx + r, sy + r)
        self.fcanvas.coords(lbl, sx, sy - r - 10)

        hand_name = 'R' if hand_idx == 0 else 'L'
        hover_text = f'{hand_name}: {hovered_key.replace("_R","") if hovered_key else ""}'
        self.fcanvas.itemconfig(lbl, text=hover_text)
        self.fcanvas.update_idletasks()

    def try_press(self, hand_idx):
        key = self.hovered.get(hand_idx)
        if not key:
            return

        now = time.time()
        if now - self.pressed_cooldown.get(key, 0) < self.COOLDOWN:
            return
        self.pressed_cooldown[key] = now

        press_key = self.KEY_MAP.get(key, key.lower())
        pyautogui.press(press_key)
        print(f"[KB] Pressed: {key}")

        self.flash_pending[key] = True
        self._refresh_key(key)

        def clear_flash(k=key):
            self.flash_pending.pop(k, None)
            self._refresh_key(k)

        self.win.after(200, clear_flash)

    def destroy(self):
        self.finger_overlay.destroy()
        self.win.destroy()
