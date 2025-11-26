import mmap
import os
import numpy as np
import cv2

# ================= CONFIG =================
TFT_WIDTH = 480
TFT_HEIGHT = 320
TFT_DEVICE = "/dev/fb1"

class TFTImageDisplay:
    def __init__(self):
        # เปิด Framebuffer และทำ mmap (Direct Memory Access)
        try:
            self.f = os.open(TFT_DEVICE, os.O_RDWR)
            # ขนาดไฟล์ = กว้าง x สูง x 2 bytes (16-bit color)
            self.fb = mmap.mmap(self.f, TFT_WIDTH * TFT_HEIGHT * 2)
        except Exception as e:
            print(f"Error opening framebuffer: {e}")
            exit(1)
            
    def convert_rgb565(self, image: np.ndarray) -> bytearray:
        """แปลงรูปในรูปแบบ RGB565"""

        # แยกช่องสี (PIL เป็น RGB)
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        # คำนวณ RGB565 แบบ Vectorized (ทำทีเดียวทั้งภาพ)
        # สูตร: (R & 0xF8) << 8 | (G & 0xFC) << 3 | (B >> 3)
        rgb565 = ((r.astype(np.uint16) & 0xF8) << 8) | \
                 ((g.astype(np.uint16) & 0xFC) << 3) | \
                 (b.astype(np.uint16) >> 3)

        return rgb565.tobytes()
    
    def show_image_file(self, image_path):
        """อ่านไฟล์รูป -> แปลงสี -> โชว์ขึ้นจอ"""
        try:
            img = cv2.imread(image_path)
            
            self.show_image(img)
            
        except FileNotFoundError:
            print(f"Error: ไม่พบไฟล์รูปภาพ {image_path}")
        except Exception as e:
            print(f"Error: {e}")
            
    def show_image(self, image: np.ndarray):
        """แสดงรูป"""
        
        if image.shape[1] != TFT_WIDTH or image.shape[0] != TFT_HEIGHT:
            image = cv2.resize(image, (TFT_WIDTH, TFT_HEIGHT))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        buffer = self.convert_rgb565(image)
        
        self.fb.seek(0)
        self.fb.write(buffer)
        
    def close(self):
        self.fb.close()
        os.close(self.f)
        
if __name__ == "__main__":
    display = TFTImageDisplay()
    display.show_image_file("UI_Images/navBar.jpg")
    
    # หมายเหตุ: เมื่อโปรแกรมจบ รูปจะยังค้างอยู่ที่หน้าจอ (ซึ่งเป็นเรื่องดี)
    display.close()