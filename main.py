import digitalio
import board
import cv2
import busio
import board
import sys
import os
import numpy as np
import adafruit_rgb_display.st7789 as st7789            # change _INVON to _INVOFF in st7789.py line 143 for invsert screen color    
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from gpiozero import Button, DigitalOutputDevice
from time import sleep
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from dotenv import load_dotenv, set_key
from typing import Optional, Dict

from libs.ReadTouch import TouchInput
from libs.ImageDisplay import TFTImageDisplay
from libs.measure import measure_beads_with_unpeel
from libs.detect_rectangle import detect_red_rectangles

class MainSystem:
    def __init__(self):
        self.screen = TFTImageDisplay()
        self.touch = TouchInput()
        # self.NevBar = cv2.imread("./UI_Images/navBar.jpg")
        # self.NevBar = cv2.cvtColor(self.NevBar, cv2.COLOR_BGR2RGB)
        self.top_left_ex = (5, 10)
        self.top_right_ex = (5, 70)
        self.bottom_left_ex = (42, 10)

    def Distribution_plot(self, size_conv, guideline_scale):
        width_px, height_px = 480, 320
        dpi = 100
        figsize = (width_px / dpi, height_px / dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        canvas = FigureCanvas(fig)

        size_conv = np.array(size_conv)

        # หา unique values และจำนวนที่เกิดขึ้น
        unique_vals, counts = np.unique(size_conv, return_counts=True)

        # คำนวณเปอร์เซ็นต์
        percentages = (counts / len(size_conv)) * 100

        bars = ax.bar(unique_vals.astype(str), percentages, color="#7b68ee", edgecolor="#4b0082", alpha=0.75, width=0.5)

        # แสดงเปอร์เซ็นต์บนแท่ง
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # ตำแหน่ง x
                bar.get_height() + 0.5,             # ตำแหน่ง y (สูงกว่าแท่งเล็กน้อย)
                f"{pct:.1f}",                      # แสดงค่า %
                ha='center', va='bottom', color='black', fontsize=7, fontweight='bold'
            )

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_title(f"Distribution (guideline {guideline_scale} mm.)", fontsize=10, weight='bold', color="#333333")
        ax.set_xlabel("Size mm.", fontsize=10, color="#555555")
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.set_ylabel("Frequency (%)", fontsize=10, color="#555555")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        plt.tight_layout()

        # วาด canvas
        canvas.draw()

        # ใช้ buffer_rgba() → แปลงเป็น numpy array
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)

        return buf[:, :, :3]
    
    def handle_scroll(self):
        view_h = 320

        touch_data = (self.touch.x, self.touch.y)
        if touch_data:
            raw_x, _ = touch_data

            if self.last_touch_x is not None:
                # dx = การเปลี่ยนตำแหน่งนิ้วแนวตั้ง แต่จอจะมองเป็นแนวนอน จึงต้องใช้ x
                dx = raw_x - self.last_touch_x

                # ทิศทางถูก: ลากลง -> scroll_x ลด
                self.scroll_x -= dx

                # จำกัดไม่ให้เกินขอบ
                self.scroll_x = max(0, min(self.full_h - view_h, self.scroll_x))

            self.last_touch_x = raw_x
        else:
            self.last_touch_x = None

        # ตัดภาพตาม scroll_x
        view = self.full_img[self.scroll_x:self.scroll_x + view_h, :, :]
        img = Image.fromarray(view)
        self.screen.show_image(img)

    def manage_scroll(self, full_img: np.ndarray):
        self.last_touch_x = None
        self.scroll_x = 0
        self.full_h, self.full_w, _ = full_img.shape
        self.full_img = full_img
        
        while True:
            if self.scroll_x <= 20:
                touch_data = self.touch.xpt.get_raw_touch()
                if touch_data:
                    x, y = touch_data
                    if (x >= self.top_left_ex[1] and x <= self.top_right_ex[1] and
                        y >= self.top_left_ex[0] and y <= self.bottom_left_ex[0]):
                        break
                    
            self.handle_scroll()

    def num_pad(self, touch_x: int, touch_y: int) -> :
        """
        ฟังก์ชันสำหรับตรวจสอบว่าพิกัด (touch_x, touch_y) อยู่ในสี่เหลี่ยมอันไหน
        ในตารางขนาด 3x3
        """

        # --- 1. กำหนดค่าพื้นฐานของตารางและสี่เหลี่ยม ---
        
        start_xy = (12, 62)                     # พิกัดเริ่มต้นของสี่เหลี่ยมอันแรก (มุมบนซ้าย)
        rect_size = (70, 35)                    # ขนาดของสี่เหลี่ยมแต่ละอัน
        horizontal_gap = 5                      # ระยะห่างระหว่างสี่เหลี่ยม
        vertical_gap = 6
        table = (3, 3)                          # จำนวนแถวและคอลัมน์
        zero_start = (12, 185)                  # พิกัดปุ่ม 0
        zero_end = (157, 220)
        dot_start = (162, 185)                  # พิกัดปุ่ม dot
        dot_end = (232, 220)
        enter_start = (236, 144)
        enter_end = (307, 220)
        del_start = (236, 62)
        del_end = (307, 138)
        
        # --- 2. วนลูปเพื่อตรวจสอบสี่เหลี่ยมแต่ละอัน ---
        
        if (zero_start[0] <= touch_x <= zero_end[0]) and (zero_start[1] <= touch_y <= zero_end[1]):
            return "0"
        elif (dot_start[0] <= touch_x <= dot_end[0]) and (dot_start[1] <= touch_y <= dot_end[1]): 
            return "."
        elif (enter_start[0] <= touch_x <= enter_end[0]) and (enter_start[1] <= touch_y <= enter_end[1]): 
            return "enter"
        elif(del_start[0] <= touch_x <= del_end[0]) and (del_start[1] <= touch_y <= del_end[1]): 
            return "x"
        else:
            for row in range(table[0]):
                for col in range(table[1]):
                    
                    # คำนวณขอบเขตของสี่เหลี่ยมปัจจุบัน
                    rect_x_start = start_xy[0] + col * (rect_size[0] + horizontal_gap)
                    rect_x_end = rect_x_start + rect_size[0]
                    
                    rect_y_start = start_xy[1] + row * (rect_size[1] + vertical_gap)
                    rect_y_end = rect_y_start + rect_size[1]
                    
                    # ตรวจสอบว่าจุดที่สัมผัสอยู่ในขอบเขตของสี่เหลี่ยมนี้หรือไม่
                    if (rect_x_start <= touch_x <= rect_x_end) and \
                    (rect_y_start <= touch_y <= rect_y_end):
                        
                        # คำนวณหมายเลขของสี่เหลี่ยม (เริ่มจาก 1)
                        rect_number = (row * table[1]) + col + 1
                        return str(rect_number)
                    
        # ถ้าวนลูปจนครบแล้วยังไม่เจอ แสดงว่าไม่ได้สัมผัสในสี่เหลี่ยมใดเลย
        return None
    
    def show_num_screen_input(self, screen_input):
        background_image = Image.open("./UI_Images/EPS_Guideline.png")
        font = ImageFont.truetype("DejaVuSans.ttf", 22)

        # --- กำหนดพื้นที่สำหรับแสดงตัวหนังสือ ---
        text_update_area = (130, 20)
        draw_on_frame = ImageDraw.Draw(background_image)
        
        # วาดข้อความใหม่ลงไปบนสำเนาของรูปภาพพื้นหลัง
        draw_on_frame.text(
            (text_update_area[0], text_update_area[1]), # ตำแหน่ง x, y เริ่มต้น
            screen_input,
            font=font,
            fill=(255, 255, 255) #
        )

        MainSys.screen.show_image(background_image)
    
    def sizing(self, guideline_scale: float = 5):
        cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            raise RuntimeError("ไม่สามารถเปิดกล้องได้")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ไม่สามารถอ่าน frame ได้")

            frame_measured = frame.copy()
            frame_screen, mean_size, box = detect_red_rectangles(image = frame)
            
            # แปลง BGR -> RGB และเป็น PIL Image
            frame_screen = cv2.resize(frame_screen, (320, 240))
            frame_screen = cv2.cvtColor(frame_screen, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_screen)
            self.screen.show_image(img)

            capture = self.touch.xpt.get_raw_touch()
            if capture:
                cap.release()

                pixel_mm = float(mean_size / guideline_scale)
                df, ovs = measure_beads_with_unpeel(
                            frame_measured,
                            pixel_mm = pixel_mm,
                            dedup_center_dist_frac = 0.5,
                            r_hint_px = 20,                     # ถ้ารู้คร่าว ๆ ใส่ได้ เช่น 10
                        )

                dist_plot = self.Distribution_plot(df['equiv_diam_um'], guideline_scale)
                detected_img = ovs.copy()
                cv2.drawContours(detected_img, [box], 0, (0, 255, 0), 1)
                detected_img = cv2.resize(detected_img, (320, 240))
                detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                combined_result = np.vstack((self.NevBar, detected_img, dist_plot))
                # cv2.imwrite("combined_result.jpg", cv2.cvtColor(combined_result, cv2.COLOR_RGB2BGR))
                
                self.manage_scroll(combined_result)
                self.screen.show_image_file("./UI_Images/EPS.png")
                break
        
if __name__ == "__main__":
    try:
        print("Initialize system.")
        
        dc = digitalio.DigitalInOut(board.D22)
        reset = digitalio.DigitalInOut(board.D27)
        cs = digitalio.DigitalInOut(board.D5)
        T_cs = DigitalOutputDevice(16,active_high=False,initial_value=None)
        T_clk = board.SCLK_1		
        T_mosi = board.MOSI_1	
        T_miso = board.MISO_1	
        T_irq = Button(26)
        
        current_directory = os.getcwd()
        dotenv_path = os.path.join(current_directory, 'guideline-scale/.env')
        load_dotenv(dotenv_path=dotenv_path)
        
        # Capture coord
        top_left = (97, 18)
        top_right = (97, 310)
        bottom_left = (209, 18)
        
        guide_top_left = (15, 245)
        guide_bottom_left = (64, 245)
        guide_top_right = (15, 320)
        
        MainSys= MainSystem(cs, dc, reset, T_cs, T_clk, T_mosi, T_miso, T_irq)

        MainSys.screen.show_image_file("./UI_Images/EPS.png")

        while True:
            touch_data = MainSys.touch.xpt.get_raw_touch()
            if touch_data:
                x, y = touch_data
                if x >= top_left[0] and y >= top_left[1] and x <= bottom_left[0] and y <= top_right[1]:
                    guideline_scale = float(os.getenv("guideline_scale"))
                    MainSys.sizing(guideline_scale)
                    touch_data = None
                elif x >= guide_top_left[0] and y >= guide_top_left[1] and x <= guide_bottom_left[0] and y <= guide_top_right[1]:
                    MainSys.screen.show_image_file("./UI_Images/EPS_Guideline.png")
                    screen_input = ""
                    sleep(0.3)
                    
                    while True:
                        touch_data = MainSys.touch.xpt.get_raw_touch()
                        if touch_data:
                            y, x = touch_data   # สลับตำแหน่งเพราะจอหมุน
                            touch_cap = MainSys.num_pad(x, y)
                            
                            if touch_cap:
                                if touch_cap == "x":
                                    screen_input = ""
                                elif touch_cap == "enter":
                                    if screen_input:
                                        os.environ["guideline_scale"] = screen_input
                                        set_key(dotenv_path, "guideline_scale", screen_input)
                                        MainSys.sizing(float(os.environ["guideline_scale"]))
                                        touch_data = None
                                        break
                                    sys.exit(0)
                                else:
                                    screen_input += str(touch_cap)
                                    
                                MainSys.show_num_screen_input(screen_input)
                            sleep(0.5)

    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)