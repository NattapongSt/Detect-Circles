from time import time
from threading import Thread
from evdev import InputDevice, ecodes, list_devices
from typing import Optional, Dict

"""
    สำหรับอ่านตำแหน่งที่แตะบนจอขนาด 480x320
    มีการทำ calibration เพื่อให้ตำแหน่งถูกต้อง
"""

TFT_WIDTH = 480
TFT_HEIGHT = 320

CALIBRATION = {
    'x_min': 200, 'x_max': 3900,
    'y_min': 200, 'y_max': 3800,
    'swap_xy': True,
    'invert_x': True,
    'invert_y': False
}

class TouchInput:
    def __init__(self, 
                calibration: Optional[Dict] = None,
                debug: Optional[bool] = False):
        self.cal = calibration
        self.debug = debug
        self.x = 0
        self.y = 0
        self.updated = False # flag บอกว่ามีข้อมูลใหม่
        self.device = self._find_device()
        self.last_touch_time = 0
        self.is_pressed = False
        
        if self.cal is None:
            self.cal = CALIBRATION
            
        self.start()

    def _find_device(self):
        """
        หา touch driver
        """
        try:
            for path in list_devices():
                dev = InputDevice(path)
                if any(k in dev.name.lower() for k in ['touch', 'ads7846', 'xpt', 'ili']):
                    dev.grab()
                    return dev
        except: pass
        return None

    def _map(self, v, in_min, in_max, out_min, out_max):
        """
        map ต่ำแหน่งที่ได้
        """
        return (v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def start(self):
        """
        เริ่มใช้งานจาก driver ที่เจอ
        """
        if not self.device: return
        Thread(target=self._run, daemon=True).start()

    def _run(self):
        raw_x = None
        raw_y = None
        
        # Loop อ่านค่า Touch
        for event in self.device.read_loop():
            # เก็บค่า Raw ไว้ก่อนเสมอ (ยังไม่คำนวณ)
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X: raw_x = event.value
                elif event.code == ecodes.ABS_Y: raw_y = event.value
            
            # เช็คสถานะการกด (กดค้าง = 1, ปล่อย = 0)
            elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TOUCH:
                self.is_pressed = (event.value == 1)

            # คำนวณพิกัดเมื่อจบแพ็คเกจ (EV_SYN) เท่านั้น
            # เพื่อให้มั่นใจว่าได้ทั้ง X และ Y ของรอบนั้นๆ ครบแล้ว
            elif event.type == ecodes.EV_SYN:
                
                # ถ้ายังไม่มีค่า X, Y ให้ข้าม
                if raw_x is None or raw_y is None:
                    continue

                # ถ้ามีการกดอยู่ -> ให้คำนวณพิกัดตลอดเวลา (Real-time update)
                if self.is_pressed:
                    # คำนวณ Calibration
                    if self.cal['swap_xy']: tx, ty = raw_y, raw_x
                    else: tx, ty = raw_x, raw_y

                    px = self._map(tx, self.cal['x_min'], self.cal['x_max'], 0, TFT_WIDTH)
                    py = self._map(ty, self.cal['y_min'], self.cal['y_max'], 0, TFT_HEIGHT)

                    if self.cal['invert_x']: px = TFT_WIDTH - px
                    if self.cal['invert_y']: py = TFT_HEIGHT - py

                    # [จุดสำคัญ] อัปเดตพิกัด self.x/y ทันทีที่นิ้วขยับ
                    self.x = int(max(0, min(TFT_WIDTH, px)))
                    self.y = int(max(0, min(TFT_HEIGHT, py)))

                    # --- Debounce Logic (สำหรับแจ้งเตือน Main Loop) ---
                    # Logic: แจ้งเตือนแค่ครั้งเดียวตอนเริ่มกดใหม่ๆ
                    current_time = time()
                    if current_time - self.last_touch_time > 0.3:
                        self.updated = True
                        self.last_touch_time = current_time
                    
                    if self.debug:
                        print(f"Touch at: ({self.x}, {self.y})")
                
    def get_current_touch(self) -> Optional[tuple[int, int]]:
        """ดึงค่าพิกัดปัจจุบัน (x, y) มีหน่วงเวลาป้องกัน debounce"""
        if self.updated:
            self.updated = False
            return (self.x, self.y)
        return None
    
    def get_touch_state(self):
        """
        คืนค่าสถานะการสัมผัสแบบ Real-time
        Return: (x, y, is_pressed)
        """
        # คืนค่าตำแหน่งปัจจุบัน และสถานะการกด (True/False)
        return (self.x, self.y, self.is_pressed)
                
if __name__ == "__main__":
    
    try:
        touch = TouchInput(debug=True)
        count = 0
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStop")