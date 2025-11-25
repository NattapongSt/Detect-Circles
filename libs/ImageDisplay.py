import mmap
import os
from PIL import Image
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
            
    def convert_rgb565(self, image: Image.Image) -> bytearray:
        """แปลงรูป PIL เป็น Byte Array ในรูปแบบ RGB565"""
        img = image.convert('RGB')
        img_np = np.array(img)

        # แยกช่องสี (PIL เป็น RGB)
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]

        # คำนวณ RGB565 แบบ Vectorized (ทำทีเดียวทั้งภาพ)
        # สูตร: (R & 0xF8) << 8 | (G & 0xFC) << 3 | (B >> 3)
        rgb565 = ((r.astype(np.uint16) & 0xF8) << 8) | \
                 ((g.astype(np.uint16) & 0xFC) << 3) | \
                 (b.astype(np.uint16) >> 3)

        return rgb565.tobytes()
    
    def show_image_file(self, image_path):
        """อ่านไฟล์รูป -> ปรับขนาด -> แปลงสี -> โชว์ขึ้นจอ"""
        try:
            img = Image.open(image_path)
            
            # แปลงเป็น RGB ธรรมดา
            img = img.convert('RGB')
            
            self.show_image(img)
            
        except FileNotFoundError:
            print(f"Error: ไม่พบไฟล์รูปภาพ {image_path}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if 'img' in locals():
                img.close()
            
    def show_image(self, image: Image.Image):
        """แสดงรูป PIL Image โดยตรง"""
        buffer = self.convert_rgb565(image)
        
        self.fb.seek(0)
        self.fb.write(buffer)

    def show_cv2_frame(self, frame):
        """
        แสดงภาพจาก OpenCV (BGR) ขึ้นจอทันที
        """
        # Resize ให้ตรงกับขนาดจอ
        if frame.shape[1] != TFT_WIDTH or frame.shape[0] != TFT_HEIGHT:
            frame = cv2.resize(frame, (TFT_WIDTH, TFT_HEIGHT))
            
        # แยกช่องสี (OpenCV เป็น BGR)
        # ใช้ Numpy Slicing
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]

        # คำนวณ RGB565 (สูตร: R5 G6 B5)
        # ใช้ .astype(np.uint16) เพื่อกันค่าล้น
        rgb565 = ((r.astype(np.uint16) & 0xF8) << 8) | \
                 ((g.astype(np.uint16) & 0xFC) << 3) | \
                 (b.astype(np.uint16) >> 3)

        # 4. เขียนลง Framebuffer (สลับ Byte ตาม Endian ของจอ)
        self.fb.seek(0)
        self.fb.write(rgb565.tobytes())
        
    def close(self):
        self.fb.close()
        os.close(self.f)
        
if __name__ == "__main__":
    display = TFTImageDisplay()
    display.show_image_file("UI_Images/navBar.jpg")
    
    # หมายเหตุ: เมื่อโปรแกรมจบ รูปจะยังค้างอยู่ที่หน้าจอ (ซึ่งเป็นเรื่องดี)
    display.close()