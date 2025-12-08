import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from dotenv import load_dotenv, set_key
from typing import Optional

from libs.ReadTouch import TouchInput
from libs.ImageDisplay import TFTImageDisplay
from libs.measure import EPS_Measurement
from libs.detect_rectangle import detect_red_rectangles

class MainSystem:
    def __init__(self):
        self.screen = TFTImageDisplay()
        self.touch = TouchInput()
        self.NevBar = cv2.imread("UI_Images/navBar.jpg", cv2.IMREAD_COLOR_RGB)
        self.top_left_ex = (20, 7)
        self.top_right_ex = (80, 7)
        self.bottom_left_ex = (20, 42)
    
    def Distribution_plot(self, 
                        size_conv: list[float], 
                        guideline_scale: float = 5) -> np.ndarray:
        """
        create distribution graph 

        Args:
            size_conv (list[float]): detected size
            guideline_scale (float, optional): Sticker size guideline. Defaults to 5.

        Returns:
            np.ndarray: distribution graph image
        """
        
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
        ax.set_title(f"Distribution (guideline {guideline_scale} mm.)", fontsize=12, weight='bold', color="#333333")
        ax.set_xlabel("Size mm.", fontsize=12, color="#555555")
        ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax.set_ylabel("Frequency (%)", fontsize=12, color="#555555")
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
        plt.close()
        return buf[:, :, :3]
    
    def handle_scroll(self, touch_state: tuple[int, int, bool]):
        """
        scroll display

        Args:
            touch_state (tuple[int, int, bool]): state touch x, y, bool (True=touch)
        """
        view_h = 320
    
        x, y, is_pressed = touch_state
        if is_pressed:
            if self.last_touch_y is not None:
                delta = y - self.last_touch_y

                # ถ้าค่าเปลี่ยนแปลงน้อยเกินไป (Noise) ให้ข้ามไป
                if abs(delta) > 2: 
                    self.scroll_y -= delta # ถ้าทิศทางกลับด้าน ให้เปลี่ยนเป็น +=
                    self.scroll_y = max(0, min(self.full_h - view_h, self.scroll_y))
                    self.last_touch_y = y # อัปเดตตำแหน่งล่าสุด

            self.last_touch_y = y
        else:
            self.last_touch_y = None

        y_start = int(self.scroll_y)
        y_end = min(y_start + view_h, self.full_h)
        view = self.full_img[y_start:y_end, :, :]
        
        self.screen.show_image(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))

    def manage_scroll(self, full_img: np.ndarray):
        """_summary_

        Args:
            full_img (np.ndarray): nav, detected, distribution graph image
        """
        self.last_touch_y = None
        self.scroll_y = 0
        self.full_h, self.full_w, _ = full_img.shape
        self.full_img = full_img
        
        while True:
            touch_state = self.touch.get_touch_state()
            x, y, is_pressed = touch_state
            
            # ตรวจจับกด exit -> break -> home UI
            if is_pressed and self.last_touch_y is None and self.scroll_y <= 25:
                if (x >= self.top_left_ex[0] and y >= self.top_left_ex[1] and
                    x <= self.top_right_ex[0] and y <= self.bottom_left_ex[1]):
                    break
            
            self.handle_scroll(touch_state)

    def num_pad(self, touch_x: int, 
                touch_y: int) -> Optional[str]:
        """
        ฟังก์ชันสำหรับตรวจสอบว่าพิกัด (touch_x, touch_y) อยู่ในสี่เหลี่ยมอันไหน
        
        Args:
            touch_x (int)
            touch_y (int)

        Returns:
            Optional[str]: ปุ่มที่แตะ
        """

        # --- กำหนดค่าพื้นฐานของตารางและสี่เหลี่ยม ---
        
        start_xy = (31, 76)                     # พิกัดเริ่มต้นของสี่เหลี่ยมอันแรก (มุมบนซ้าย)
        rect_size = (96, 50)                    # ขนาดของสี่เหลี่ยมแต่ละอัน (กว้าง, ยาว)
        horizontal_gap = 10                      # ระยะห่างระหว่างสี่เหลี่ยม
        vertical_gap = 5
        table = (3, 3)                          # จำนวนแถวและคอลัมน์
        zero_start = (31, 242)                  # พิกัดปุ่ม 0
        zero_end = (232, 292)
        dot_start = (242, 242)                  # พิกัดปุ่ม dot
        dot_end = (337, 292)
        enter_start = (347, 187)
        enter_end = (448, 292)
        del_start = (347, 76)
        del_end = (448, 182)
        
        # --- วนลูปเพื่อตรวจสอบสี่เหลี่ยมแต่ละอัน ---
        
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
    
    def show_num_screen_input(self, screen_input: str):
        """
        display number

        Args:
            screen_input (str)
        """
        background_image = cv2.imread("UI_Images/EPS_Guideline.png")

        x, y = 195, 25

        cv2.putText(
            background_image,
            screen_input,
            (x, y + 28),               # y+size แก้ baseline
            cv2.FONT_HERSHEY_SIMPLEX,  # ฟอนต์ของ OpenCV
            1,                         # ขนาดตัวอักษร
            (255, 255, 255),           # สี (BGR)
            2,                         # ความหนา
            cv2.LINE_AA
        )
        
        MainSys.screen.show_image(background_image)
    
    def sizing(self, guideline_scale: float = 5):
        """
        เริ่มใช้งาน
        เปิดกล้องและวัดขนาดสติกเกอร์ guideline 
        รอแตะที่จอ เมื่อแตะที่จอจะไปวัดขนาดเม็ดพลาสติก

        Args:
            guideline_scale (float, optional): _description_. Defaults to 5.

        Raises:
            RuntimeError: _description_
        """
        
        eps = EPS_Measurement()
        cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            raise RuntimeError("ไม่สามารถเปิดกล้องได้")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ไม่สามารถอ่าน frame ได้")

            frame_screen, mean_size, box = detect_red_rectangles(image = frame)
            
            self.screen.show_image(frame_screen)

            capture = self.touch.get_current_touch()
            if capture:
                
                cap.release()
                pixel_mm = mean_size / guideline_scale
                df, ovs = eps.measure_beads_with_unpeel(
                            frame,
                            box=box,
                            pixel_mm = pixel_mm,
                            dedup_center_dist_frac = 0.5,
                            r_hint_px = 20,                     # ถ้ารู้คร่าว ๆ ใส่ได้ เช่น 10
                        )
                
                dist_plot = self.Distribution_plot(df['equiv_diam_um'], guideline_scale)
                
                detected_img = ovs.copy()
                cv2.drawContours(detected_img, [box], 0, (0, 255, 0), 1)
                detected_img = cv2.resize(detected_img, (480, 320))
                detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                
                combined_result = np.vstack((self.NevBar, detected_img, dist_plot))
                # cv2.imwrite("combined_result.jpg", cv2.cvtColor(combined_result, cv2.COLOR_RGB2BGR))
        
                self.manage_scroll(combined_result)
                self.screen.show_image_file("UI_Images/EPS.png")
                break
        
if __name__ == "__main__":
    try:
        print("Initialize system.")
        dotenv_path = "guideline-scale/.env"
        load_dotenv(dotenv_path=dotenv_path)
        
        # Capture coord
        top_left = (20, 118)
        top_right = (460, 118)
        bottom_left = (20, 293)
        
        guide_top_left = (351, 17)
        guide_bottom_left = (351, 77)
        guide_top_right = (480, 17)
        
        MainSys= MainSystem()
        MainSys.screen.show_image_file("UI_Images/EPS.png")
        
        while True:
            touch_data = MainSys.touch.get_current_touch()              # รออ่านตำแน่งที่แตะ
            if touch_data:
                x, y = touch_data
                if x >= top_left[0] and y >= top_left[1] and x <= top_right[0] and y <= bottom_left[1]:
                    """Sizing area touched"""
                    
                    guideline_scale = float(os.getenv("guideline_scale"))
                    MainSys.sizing(guideline_scale)
                    touch_data = None
                elif x >= guide_top_left[0] and y >= guide_top_left[1] and x <= guide_top_right[0] and y <= guide_bottom_left[1]:
                    """Guideline area touched"""
                    
                    MainSys.screen.show_image_file("UI_Images/EPS_Guideline.png")
                    screen_input = ""
                    
                    while True:
                        touch_data = MainSys.touch.get_current_touch()
                        if touch_data:
                            x, y = touch_data
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
                                    pass
                                else:
                                    screen_input += str(touch_cap)
                                    
                                MainSys.show_num_screen_input(screen_input)
                        
                        sleep(0.05)

    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)