import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class EPS_Measurement:
    """
    EPS size measurement from the edge of the circle
    Main steps
    - Divide the detection into small and large particles due to different parameter settings
    - Image preparation steps
    - Union of all results
    
    cv2.HoughCircles()
    cv2.HOUGH_GRADIENT      วิธีการตรวจจับวงกลม
                            ใช้ Gradient-based Hough Transform
    dp                      Inverse ratio of accumulator resolution
                            ค่าความละเอียดของตัว accumulator ที่ใช้หา circle
                            dp = 1 → ความละเอียดเท่ากับภาพจริง
    minDist                 ระยะห่างขั้นต่ำระหว่างศูนย์กลางของวงกลมแต่ละอัน
                            ถ้าตั้งน้อยเกิน → เจอวงกลมซ้อน
                            ถ้าตั้งมากเกิน → อาจไม่เจอบางวง
    param1                  ค่าขอบเขตสำหรับ Canny edge detector (upper threshold)
                            ใช้สร้างขอบให้ Hough เจอวงกลมได้ดีขึ้น
    param2                  ค่าที่คุมความ “เข้มงวด” ในการตรวจจับวงกลม
                            ค่าต่ำ → เจอวงกลมเยอะ (รวม fake)
                            ค่าสูง → เจอวงกลมน้อยแต่แม่น
    minRadius               รัศมี ขั้นต่ำ ของวงกลมที่ต้องการหา
    maxRadius               รัศมี สูงสุด ของวงกลมที่จะค้นหา
    """
    
    def __init__(self):
        """
        initial variables
        """
        self._cache_top = None
        self._cache_bottom = None
        self.img_bgr = None
        self.img_gray = None
        self.box = None
        self.r_hint_px = None
        
    def _nms_merge_circles(self,
                           primary_cir: List[Tuple[float, float, float]],
                           secound_cir: List[Tuple[float, float, float]],
                           min_center_dist_frac: float = 0.5) -> List[Tuple[float, float, float]]:
        """
        รวมผลตรวจจับจากสองรอบ (primary_cir ก่อน secound_cir) และกันซ้ำแบบง่าย:
        ถ้าศูนย์กลางห่างกัน < min_center_dist_frac*(r1+r2) → ถือว่าเป็นเม็ดเดียวกัน ใช้ของ primary_cir

        Args:
            primary_cir (List[Tuple[float, float, float]]): วงกลม primary ที่จะถูกเลือกหากซ้อน
            secound_cir (List[Tuple[float, float, float]]): วงกลม secoundary
            min_center_dist_frac (float, optional): ระยะห่างของจุดศูนย์กลาง Defaults to 0.5.

        Returns:
            List[Tuple[float, float, float]]: list ของวงกลมหลังจาก merge
        """
        
        out = primary_cir[:]
        for (x2,y2,r2) in secound_cir:
            duplicate = False
            for (x1,y1,r1) in primary_cir:
                d = ((x1-x2)**2 + (y1-y2)**2) ** 0.5
                if d < min_center_dist_frac*(r1+r2):
                    duplicate = True
                    break
            if not duplicate:
                out.append((x2,y2,r2))
        return out
    
    def _circles_to_df(self,
                       circles: List[Tuple[float, float, float]],
                       pixel_size_um: Optional[float]) -> pd.DataFrame:
        """
        Create dataframe for all circles

        Args:
            circles (List[Tuple[float, float, float]]): circles
            pixel_size_um (Optional[float]): pixel per millimeters

        Returns:
            pd.DataFrame: result circles
        """
        
        rows = []
        for index, (x, y, r) in enumerate(circles, start=1):
            row = {
                "id": index,                                # id
                "cx_px": float(x),                          # center point x
                "cy_px": float(y),                          # center point y
                "radius_px": float(r),                      # radius of circle
                "equiv_diam_px": float(2*r),                # Diameter of circle
                "area_px2": float(np.pi*(r**2)),            # Area of ​​circle
                "perimeter_px": float(2*np.pi*r),           # perimeters of circle
            }
            
            if pixel_size_um is not None:
                row["radius_px"] = round(row["radius_px"] / pixel_size_um, 1)
                equiv_diam_px = round(row["equiv_diam_px"] / pixel_size_um, 2)
                
                # check if out of 10% of boundary
                if 0.6 <= equiv_diam_px <= 1.28:
                    # Polynomial Degree 2
                    # rescaled = (-1.00420447 * equiv_diam_px**2) + (3.2192789 * equiv_diam_px) -1.0031286
                    rescaled = (-0.95526892 * equiv_diam_px**2) + (3.12504613 * equiv_diam_px) -0.96687283
                    row["equiv_diam_um"] = round(rescaled, 1)
                else:
                    # raw value
                    row["equiv_diam_um"] = round(equiv_diam_px, 1)
                    
                # row["equiv_diam_umr2"] = round(row["equiv_diam_px"] / pixel_size_um, 2)
            
            rows.append(row)
        return pd.DataFrame(rows)
    
    def _draw_circles(self,
                      overlay: np.ndarray,
                      circles: List[Tuple[float, float, float]],
                      color: Tuple[int, int, int],
                      thick: int = 1) -> np.ndarray:
        """
        draw circles

        Args:
            overlay (np.ndarray): image
            circles (List[Tuple[float, float, float]]): list circles
            color (Tuple[int, int, int]): color BGR
            thick (int, optional): thickness Defaults to 1.

        Returns:
            np.ndarray: circles image
        """
        
        o = overlay.copy()
        for (x, y, r) in circles:
            cv2.circle(o, (int(x), int(y)), int(r), color, thick)
            cv2.circle(o, (int(x), int(y)), 1, (255, 255, 255), -1)
        
        return o

    def preprocess_img(self,
                       img_gray: np.ndarray,
                       cliLimit: float,
                       titleGridSize: Tuple[int, int],
                       is_sharp: bool = False,
                       median_blur_ksize: int = 3) -> np.ndarray:
        """
        Clean up the image to remove noise and increase sharpness.

        Args:
            img_gray (np.ndarray): gray image
            cliLimit (float): sets the threshold for contrast limiting
            titleGridSize (Tuple[int, int]): This parameter is a tuple of two integers, representing the (width, height) of the grid for dividing the image into tiles
            is_sharp (bool, optional): Defaults to False.
            median_blur_ksize (int, optional): the kernel size. It defines the size of the square neighborhood. Defaults to 3.

        Returns:
            np.ndarray: result preprocess image
        """
        
        # --- CLANE เพื่อเพิ่มคอนทราสต์ ---
        clahe = cv2.createCLAHE(clipLimit=cliLimit, tileGridSize=titleGridSize)
        enhanced = clahe.apply(img_gray)
        
        if is_sharp:
            """
            ทำให้ขอบชัดขึ้นด้วยเทคนิค Sharpening ด้วย Unsharp Masking
            ภาพจริง - ภาพเบลอ
            """
            blur = cv2.bilateralFilter(enhanced, 9, 100, 100)
            enhanced = cv2.addWeighted(enhanced, 2.5, blur, -1.5, 0)
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
        # --- Median blur เล็กน้อยเพื่อลด noise ---
        return cv2.medianBlur(enhanced, median_blur_ksize)
    
    def filter_circles(self,
                       circles: List) -> List[Tuple[float, float, float]]:
        """
        Filter out circles in the guideline.

        Args:
            circles (List): list circles

        Returns:
            List[Tuple[float, float, float]]: Circle after filtering
        """
        
        filtered = []
        box_polygon = self.box.reshape((-1, 1, 2))
        
        for c in circles[0]:
            center = (np.float32(c[0]), np.float32(c[1]))
            
            # ตรวจสอบว่าจุดศูนย์กลางอยู่ในสีเหลี่ยมไหม
            if cv2.pointPolygonTest(box_polygon, center, False) < 0:
                filtered.append(tuple(map(float, c)))
                
        return filtered
    
    def inpaint_circles(self,
                    gray: np.ndarray,
                    circles: List[Tuple[float, float, float]],
                    expand: float = 1.05) -> np.ndarray:
        """
        inpaint detected circles

        Args:
            gray (np.ndarray): gray image
            circles (List[Tuple[float, float, float]]): list circles
            expand (float, optional): Defaults to 1.05.

        Returns:
            np.ndarray: result image
        """
        
        mask = np.zeros_like(gray, np.uint8)
        for (x, y, r) in circles:
            cv2.circle(mask, (int(x), int(y)), int(r*expand), 255, -1)
        return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    
    def inpaint_guideline(self,
                          gray: np.ndarray,
                          expand: float = 1.05) -> np.ndarray:
        """
        สร้าง inpaint mask จากพิกัด 4 จุดมุม (Polygon/Rotated Rect) แล้วลบออก

        Args:
            gray (np.ndarray): ภาพ grayscale input
            expand (float, optional): ตัวคูณขยายขนาด (1.05 = ขยาย 5%)

        Returns:
            np.ndarray: รูปที่ลบ guideline
        """
        mask = np.zeros_like(gray, np.uint8)
        pts = np.array(self.box, dtype=np.float32)
        
        center = np.mean(pts, axis=0)                   # หาจุดกึ่งกลาง (Centroid) ของสี่เหลี่ยม
        vectors = pts - center                          # ขยายจุดทั้ง 4 ออกจากจุดกึ่งกลาง
        expanded_pts = center + (vectors * expand)      # สูตร: จุดใหม่ = จุดกึ่งกลาง + (ระยะห่างจากกลาง * ตัวคูณ)
        final_pts = expanded_pts.astype(np.int32)       # แปลงกลับเป็น int เพื่อวาดลง Mask
        cv2.fillPoly(mask, [final_pts], 255)            # วาด Polygon สีขาวลงบน Mask
        
        # Inpaint
        return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    
    def inpaint_circles_batch(self, 
                              gray: np.ndarray, 
                              circles_list: List[List[Tuple[float, float, float]]],
                              expand: float = 1.05) -> np.ndarray:
        """
        inpaint detected circles (for many layer)

        Args:
            gray (np.ndarray): gray image
            circles_list (List[List[Tuple[float, float, float]]]): List of circles for each layer
            expand (float, optional): Defaults to 1.05.

        Returns:
            np.ndarray: result image
        """
        
        mask = np.zeros_like(gray, np.uint8)
        for circles in circles_list:
            for (x,y,r) in circles:
                cv2.circle(mask, (int(x),int(y)), int(r*expand), 255, -1)
        return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    
    def detect_top_layer(self,
                         minR: Optional[int] = 10,
                         maxR: Optional[int] = 40,
                         param1: int = 95,
                         param2: int = 30,
                         min_dist_factor: float = 0.9) -> List[Tuple[float,float,float]]:
        """
        detect beads top layer

        Args:
            minR (Optional[int], optional): min radius of circles. Defaults to 10.
            maxR (Optional[int], optional): max radius of circles. Defaults to 40.
            param1 (int, optional): parameters the Canny edge detector. Defaults to 95.
            param2 (int, optional): it is the accumulator threshold for the circle centered at the detection stage. Defaults to 30.
            min_dist_factor (float, optional): minimum distance between the centers of the detected circles. Defaults to 0.9.

        Returns:
            List[Tuple[float,float,float]]: list detected circles
        """
        
        # debug image
        cv2.imwrite("top.png", self._cache_top)
        
        if self.r_hint_px is None:
            self.r_hint_px = max(6, min(self.img_gray.shape[:2]) / 60.0)
            
        minDist = int(max(4, self.r_hint_px*min_dist_factor))
        circles = cv2.HoughCircles(self._cache_top,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1,
                                   minDist=minDist,
                                   param1=param1,
                                   param2=param2,
                                   minRadius=minR,
                                   maxRadius=maxR)

        top = []
        if circles is not None:
            top = self.filter_circles(circles)
        return top
    
    def detect_bottom_layer_from_inpaint(self,
                                         minR: Optional[int] = 10,
                                         maxR: Optional[int] = 40,
                                         param1: int = 70,
                                         param2: int = 25,
                                         min_dist_factor: float = 0.85) -> List[Tuple[float, float, float]]:
        """
        detect beads bottom layer (The parameters should be different from top layer.)

        Args:
            minR (Optional[int], optional): min radius of circles. Defaults to 10.
            maxR (Optional[int], optional): max radius of circles. Defaults to 40.
            param1 (int, optional): parameters the Canny edge detector. Defaults to 95.
            param2 (int, optional): it is the accumulator threshold for the circle centered at the detection stage. Defaults to 30.
            min_dist_factor (float, optional): minimum distance between the centers of the detected circles. Defaults to 0.9.

        Returns:
            List[Tuple[float,float,float]]: list detected circles
        """
        
        # debug image
        cv2.imwrite("bot.png", self._cache_bottom)
        
        if self.r_hint_px is None:
            self.r_hint_px = max(6, min(self.img_bgr.shape[:2]) / 60.0)
            
        minDist = int(max(4, self.r_hint_px*min_dist_factor))
        circles = cv2.HoughCircles(self._cache_bottom,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1,
                                   minDist=minDist,
                                   param1=param1,
                                   param2=param2,
                                   minRadius=minR,
                                   maxRadius=maxR
                                   )
        
        bottom = []
        if circles is not None:
            bottom = self.filter_circles(circles)
        return bottom
    
    def measure_beads_with_unpeel(self,
                                  img_bgr: np.ndarray,
                                  box: np.ndarray,
                                  pixel_mm: float = None,
                                  r_hint_px: Optional[float] = None,
                                  dedup_center_dist_frac: float = 0.5
                                  ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        detect eps beads
         - detect lower size (top, bottom layer)
         - detect upper size (top, bottom layer)

        Args:
            img_bgr (np.ndarray): BGR image
            box (np.ndarray): Coordinates Guideline
            pixel_mm (float, optional): pixel per millimeters. Defaults to None.
            r_hint_px (Optional[float], optional): Defaults to None.
            dedup_center_dist_frac (float, optional): Defaults to 0.5.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Details of detected circles and overlay image
        """
        
        self.img_bgr = img_bgr
        self.box = box
        self.r_hint_px = r_hint_px
        
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.img_gray = self.inpaint_guideline(self.img_gray, expand=1)
        
        self._cache_top = self.preprocess_img(self.img_gray, 2, (4,4), True, 5)
        self._cache_bottom = self.preprocess_img(self.img_gray, 2, (4,4), False, 3)
        
        is_inpainted = False
        merged = {"upper_size": [], "lower_size": []}
        ov_img = self.img_bgr.copy()

        for bead in ["lower_size", "upper_size"]:
            if bead == "lower_size":
                minR = round((pixel_mm * 0.3) / 2)
                maxR = round((pixel_mm * 0.7) / 2)
                top_param2 = 30
                bottom_param2 = 27
                param1 = 80
            else:
                minR = round((pixel_mm * 0.7) / 2) + 1
                maxR = round((pixel_mm * 1.5) / 2)
                top_param2 = 35
                bottom_param2 = 30
                param1 = 90
                
            # top layer
            c_top = self.detect_top_layer(minR=minR,
                                          maxR=maxR,
                                          param1=param1,
                                          param2=top_param2)
            
            # Inpaint top layer
            self._cache_bottom = self.inpaint_circles(self._cache_bottom, c_top)
            
            # bottom layer
            c_bot = self.detect_bottom_layer_from_inpaint(minR=minR,
                                                          maxR=maxR,
                                                          param1=param1,
                                                          param2=bottom_param2)
            
            # inpaint all circles (top, bottom image)
            if not is_inpainted:
                self._cache_top = self.inpaint_circles_batch(self._cache_top, [c_top, c_bot])
                self._cache_bottom = self.inpaint_circles(self._cache_bottom, c_bot)
            
            # Merge + NMS (กันซ้ำ)
            merged[bead] = self._nms_merge_circles(c_bot, c_top, min_center_dist_frac=dedup_center_dist_frac)
            is_inpainted = True
            
        merged_all = self._nms_merge_circles(merged["upper_size"], 
                                             merged["lower_size"], 
                                             min_center_dist_frac=dedup_center_dist_frac)
        ov_img = self._draw_circles(ov_img, merged_all, color=(255, 0, 0))
        df_merged = self._circles_to_df(merged_all, pixel_size_um=pixel_mm)
        
        # size_conv = np.array(df_merged["equiv_diam_umr2"])
        # unique_vals, counts = np.unique(size_conv, return_counts=True)
        # percentages = (counts / len(size_conv)) * 100
        # print("\n")
        # for val, pct in zip(unique_vals, percentages):
        #     print(f"  {val:.2f} mm: {pct:.1f}%")
        # print("\n")
    
        return df_merged, ov_img
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from detect_rectangle import detect_red_rectangles
    from time import time
    
    load_dotenv(dotenv_path="/home/eps/EPS_Detection/guideline-scale/.env")
    
    eps = EPS_Measurement()
    img = cv2.imread("dataset/not_flash/size1.4/161059.jpg")
    frame_screen, mean_size, box = detect_red_rectangles(image=img)
    
    pixel_mm = mean_size / float(os.getenv("guideline_scale"))
    
    for i in range(2):
        start_time = time()
        
        df, ov_img = eps.measure_beads_with_unpeel(img, 
                                                box=box,
                                                pixel_mm=pixel_mm,
                                                r_hint_px=20)    
        
        cv2.imwrite("ov_img_class.png", ov_img)
        
        print(f"\nused time: {time() - start_time}")
        
    print(df)