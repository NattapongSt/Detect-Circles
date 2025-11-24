from time import sleep
from threading import Thread
from evdev import InputDevice, ecodes, list_devices
from typing import Optional, Dict

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
    def __init__(self, calibration: Optional[Dict] = None):
        self.cal = calibration
        self.x = 0
        self.y = 0
        self.updated = False # flag บอกว่ามีข้อมูลใหม่
        self.device = self._find_device()
        
        if self.cal is None:
            self.cal = CALIBRATION
            
        self.start()

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
        Thread(target=self._run, daemon=True).start()

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
                
if __name__ == "__main__":
    
    try:
        touch = TouchInput()
        
        while True:
            if touch.updated:
                # ดึงค่าล่าสุดมาวาด
                current_x = touch.x
                current_y = touch.y
                print(f"Touch at: ({current_x}, {current_y})")
                touch.updated = False # เคลียร์ flag
            
            # หน่วงเวลาเล็กน้อยเพื่อให้ CPU ไม่โหลด 100% (ประมาณ 60 FPS)
            sleep(0.016)
    except KeyboardInterrupt:
        print("\nStop")