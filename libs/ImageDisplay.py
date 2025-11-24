import mmap
import os
from PIL import Image

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
        pixels = list(img.getdata())
        buffer = bytearray(TFT_WIDTH * TFT_HEIGHT * 2)
        
        index = 0
        for r, g, b in pixels:
            val = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            buffer[index] = val & 0xFF
            buffer[index+1] = (val >> 8) & 0xFF
            index += 2
        return buffer
    
    def show_image_file(self, image_path):
        """อ่านไฟล์รูป -> ปรับขนาด -> แปลงสี -> โชว์ขึ้นจอ"""
        try:
            img = Image.open(image_path)
            
            # แปลงเป็น RGB ธรรมดา
            img = img.convert('RGB')
            
            buffer = self.convert_rgb565(img)
            
            # เขียนลงจอทีเดียว (เร็วมากเพราะใช้ mmap)
            self.fb.seek(0)
            self.fb.write(buffer)
            
        except FileNotFoundError:
            print(f"Error: ไม่พบไฟล์รูปภาพ {image_path}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            img.close()
            
    def show_img(self, image: Image.Image):
        """แสดงรูป PIL Image โดยตรง"""
        buffer = self.convert_rgb565(image)
        
        self.fb.seek(0)
        self.fb.write(buffer)

    def close(self):
        self.fb.close()
        os.close(self.f)
        
if __name__ == "__main__":
    display = TFTImageDisplay()
    display.show_image_file("UI_Images/EPS.png")
    
    # หมายเหตุ: เมื่อโปรแกรมจบ รูปจะยังค้างอยู่ที่หน้าจอ (ซึ่งเป็นเรื่องดี)
    display.close()