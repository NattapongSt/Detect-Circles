#!/usr/bin/env python3
import mmap
import os
import struct
import time
import threading
from evdev import InputDevice, ecodes, list_devices

# ================= CONFIG =================
TFT_WIDTH = 480
TFT_HEIGHT = 320
TFT_DEVICE = "/dev/fb1"

# ใส่ค่า Calibrate ที่หาได้จากขั้นตอนก่อนหน้านี้
CALIBRATION = {
    'x_min': 200, 'x_max': 3900,
    'y_min': 200, 'y_max': 3800,
    'swap_xy': True,
    'invert_x': True,
    'invert_y': False
}

# ================= CLASS: FAST DISPLAY =================
class FastDisplay:
    def __init__(self, device_path):
        self.width = TFT_WIDTH
        self.height = TFT_HEIGHT
        
        # เปิดไฟล์ Framebuffer
        self.f = os.open(device_path, os.O_RDWR)
        
        # ใช้ mmap เพื่อแปลงไฟล์เป็น RAM (เขียนปุ๊บ ภาพมาปั๊บ เร็วมาก)
        # 480 * 320 * 2 bytes (16-bit color)
        self.fb = mmap.mmap(self.f, self.width * self.height * 2)
        
        self.last_x = -1
        self.last_y = -1
        
        # สี (RGB565 format)
        self.COLOR_RED = b'\xF8\x00'   # Red
        self.COLOR_BLACK = b'\x00\x00' # Black
        self.COLOR_WHITE = b'\xFF\xFF' # White

    def clear_screen(self):
        # ถมดำทั้งจอ
        self.fb[:] = b'\x00' * (self.width * self.height * 2)

    def draw_pixel(self, x, y, color_bytes):
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 2
            self.fb[offset:offset+2] = color_bytes

    def update_cursor(self, x, y):
        """
        เทคนิค: ลบเคอร์เซอร์เก่า -> วาดเคอร์เซอร์ใหม่
        (ไม่ต้องวาด background ใหม่ทั้งจอ)
        """
        # 1. ลบตำแหน่งเก่า (ถมดำ หรือวาดภาพพื้นหลังทับจุดเดิม)
        if self.last_x != -1:
            self._draw_rect(self.last_x, self.last_y, 10, self.COLOR_BLACK) # ลบ
            
        # 2. วาดตำแหน่งใหม่
        self._draw_rect(x, y, 10, self.COLOR_RED) # วาดใหม่
        
        self.last_x = x
        self.last_y = y

    def _draw_rect(self, cx, cy, size, color):
        # วาดสี่เหลี่ยมรอบจุดศูนย์กลาง cx, cy
        half = size // 2
        for y in range(cy - half, cy + half):
            for x in range(cx - half, cx + half):
                self.draw_pixel(x, y, color)

# ================= CLASS: ASYNC TOUCH =================
class TouchInput:
    def __init__(self, calibration):
        self.cal = calibration
        self.x = 0
        self.y = 0
        self.updated = False # flag บอกว่ามีข้อมูลใหม่
        self.device = self._find_device()

    def _find_device(self):
        try:
            for path in list_devices():
                dev = InputDevice(path)
                if any(k in dev.name.lower() for k in ['touch', 'ads7846', 'xpt', 'ili']):
                    dev.grab()
                    return dev
        except: pass
        return None

    def _map(self, v, in_min, in_max, out_min, out_max):
        return (v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def start(self):
        if not self.device: return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        raw_x, raw_y = 0, 0
        
        # Loop อ่านค่า Touch
        for event in self.device.read_loop():
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X: raw_x = event.value
                elif event.code == ecodes.ABS_Y: raw_y = event.value
            
            elif event.type == ecodes.EV_SYN:
                # คำนวณพิกัดทันทีที่มี packet ครบ
                if self.cal['swap_xy']: tx, ty = raw_y, raw_x
                else: tx, ty = raw_x, raw_y

                px = self._map(tx, self.cal['x_min'], self.cal['x_max'], 0, TFT_WIDTH)
                py = self._map(ty, self.cal['y_min'], self.cal['y_max'], 0, TFT_HEIGHT)

                if self.cal['invert_x']: px = TFT_WIDTH - px
                if self.cal['invert_y']: py = TFT_HEIGHT - py

                # อัปเดตตัวแปร Global (Atomic update)
                self.x = int(max(0, min(TFT_WIDTH, px)))
                self.y = int(max(0, min(TFT_HEIGHT, py)))
                self.updated = True

# ================= MAIN LOOP =================
if __name__ == "__main__":
    try:
        display = FastDisplay(TFT_DEVICE)
        display.clear_screen()
        
        touch = TouchInput(CALIBRATION)
        touch.start()
        
        print("Start Drawing... (Ctrl+C to stop)")
        
        # Loop วาดภาพ แยกอิสระจาก Loop รับ Touch
        # เทคนิค: Frame Skipping อัตโนมัติ ถ้า Touch มาเร็วเกินไป
        # เราจะวาดแค่ค่าล่าสุดเท่านั้น
        while True:
            if touch.updated:
                # ดึงค่าล่าสุดมาวาด
                current_x = touch.x
                current_y = touch.y
                print(f"Touch at: ({current_x}, {current_y})")
                touch.updated = False # เคลียร์ flag
                
                # สั่งวาด
                display.update_cursor(current_x, current_y)
            
            # หน่วงเวลาเล็กน้อยเพื่อให้ CPU ไม่โหลด 100% (ประมาณ 60 FPS)
            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\nStop")