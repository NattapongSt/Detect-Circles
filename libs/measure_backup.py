import cv2, numpy as np, pandas as pd
from typing import List, Set, Tuple, Optional, Dict

# -------- Utils --------
def _nms_merge_circles(primary: List[Tuple[float,float,float]],
                        secondary: List[Tuple[float,float,float]],
                        min_center_dist_frac: float = 0.5) -> List[Tuple[float,float,float]]:
    """
    รวมผลตรวจจับจากสองรอบ (primary ก่อน secondary) และกันซ้ำแบบง่าย:
    ถ้าศูนย์กลางห่างกัน < min_center_dist_frac*(r1+r2) → ถือว่าเป็นเม็ดเดียวกัน ใช้ของ primary
    """

    out = primary[:]
    for (x2,y2,r2) in secondary:
        duplicate = False
        for (x1,y1,r1) in primary:
            d = ((x1-x2)**2 + (y1-y2)**2) ** 0.5
            if d < min_center_dist_frac*(r1+r2):
                duplicate = True
                break
        if not duplicate:
            out.append((x2,y2,r2))
    return out

def _circles_to_df(circles: List[Tuple[float,float,float]],
                    depth: int,
                    calibrate_scale: float,
                    pixel_size_um: Optional[float]) -> pd.DataFrame:
    
    rows = []
    for i,(x,y,r) in enumerate(circles, start=1):
        row = {
            "id": i, "cx_px": float(x), "cy_px": float(y),
            "radius_px": float(r), "equiv_diam_px": float(2*r),
            "area_px2": float(np.pi*(r**2)),
            "perimeter_px": float(2*np.pi*r),
            "depth": depth,          # 0=บน, 1=ล่าง
            "method": "hough"
        }
        if pixel_size_um is not None:
            row["radius_um"] = round(row["radius_px"] / pixel_size_um, 1)
            equiv_diam_px = round(row["equiv_diam_px"] / pixel_size_um, 2)
            
            if 0.6 <= equiv_diam_px <= 1.28:         # error 10% of bound
                # rescaled = (1.3664790494058785*equiv_diam_px) - 0.20149312070043804                           # linear
                # rescaled = (-0.99056926 * equiv_diam_px**2) + (3.19957248 * equiv_diam_px) -1.00809832           # Polynomial
                rescaled = (-1.00420447 * equiv_diam_px**2) + (3.2192789 * equiv_diam_px) -1.0031286
                # rescaled = (1.70327512 * equiv_diam_px**3) + (-5.87683002 * equiv_diam_px**2) + (7.79170769 * equiv_diam_px) - 2.41935017
                row["equiv_diam_um"] = round(rescaled, 1)
            else:
                row["equiv_diam_um"] = round(equiv_diam_px, 1)
            
            row["equiv_diam_umr2"] = round(row["equiv_diam_px"] / pixel_size_um, 2)

            # if equiv_diam_um < 0.6:
            #     row["radius_um"] = round(radius_um, 1)
            #     row["equiv_diam_um"] = round(equiv_diam_um, 1)
            # else:
            #     row["radius_um"] = round(radius_um + calibrate_scale, 1)
            #     row["equiv_diam_um"] = round(equiv_diam_um + calibrate_scale, 1)
        rows.append(row)
    return pd.DataFrame(rows)

def _draw_circles(overlay: np.ndarray,
                circles: List[Tuple[float,float,float]],
                color: Tuple[int,int,int], thick: int=1) -> np.ndarray:
    o = overlay.copy()
    for (x,y,r) in circles:
        cv2.circle(o, (int(x),int(y)), int(r), color, thick)
        cv2.circle(o, (int(x),int(y)), 1, (255,255,255), -1)
    return o

def preprocess_img(img_gray: np.ndarray,
                    cliLimit: float,
                    titleGridSize: Tuple[int,int],
                    is_filter: bool = False,
                    median_blur_ksize: int = 3,) -> np.ndarray:
    """"preprocess image เพิ่มความเข้มของสีดำและเพิ่มความคมของขอบ"""
    
    
    # --- CLAHE เพื่อเพิ่มคอนทราสต์ ---
    clahe = cv2.createCLAHE(clipLimit=cliLimit, tileGridSize=titleGridSize)
    enhanced = clahe.apply(img_gray)
    
    if is_filter:
        # blur = cv2.GaussianBlur(enhanced, (5,5), sigmaX=1)
        blur = cv2.bilateralFilter(enhanced, 9, 100, 100)
        enhanced = cv2.addWeighted(enhanced, 2.5, blur, -1.5, 0)
        
        # kernel = np.array([[0, -1,  0], 
        #                 [-1,  5, -1], 
        #                 [0, -1,  0]])
        
        # kernel = np.array([[-1, -1, -1,],
        #                    [-1, 9, -1],
        #                    [-1, -1, -1]])
        # enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
    # --- Median blur เล็กน้อยเพื่อลด noise ---
    return cv2.medianBlur(enhanced, median_blur_ksize)

def filter_circles(circles: List,
                    box: np.ndarray) -> List[Tuple[float, float, float]]:
    """Filter out the circles inside the guideline"""
    filtered = []
    box_polygon = box.reshape((-1, 1, 2))
    
    for c in circles[0]:
        center = (np.float32(c[0]), np.float32(c[1]))
        
        if cv2.pointPolygonTest(box_polygon, center, False) < 0:
            filtered.append(tuple(map(float, c)))
    
    return filtered

# -------- 1) ตรวจเม็ดชั้นบน (คม/สว่าง) --------
def detect_top_layer(inpainted_polygon: np.ndarray,
        box: np.ndarray,
        r_hint_px: Optional[float] = None,
        minR: Optional[float] = None,
        maxR: Optional[float] = None,
        param1: int = 95,
        param2: int = 22,
        min_dist_factor: float = 0.9) -> Tuple[List[Tuple[float,float,float]], np.ndarray]:
    """
    ใช้ HoughCircles ที่เข้มงวดกว่าปกติ เพื่อจับเม็ด 'ชั้นบน' ที่ขอบคม-สว่าง
    return circles (x,y,r) และ overlay ของรอบนี้
    """
    
    g = preprocess_img(inpainted_polygon, cliLimit=5, titleGridSize=(8, 8), is_filter=True, median_blur_ksize=5)
    
    cv2.imwrite("./gray_toplayer.png", g)

    if r_hint_px is None:
        r_hint_px = max(6, min(inpainted_polygon.shape[:2]) / 60.0)

    # minR = max(4, int(r_hint_px*0.6))
    # maxR = int(r_hint_px*2.5)
    minDist = int(max(4, r_hint_px*min_dist_factor))

    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
        param1=param1, param2=param2,
        minRadius=int(minR), maxRadius=int(maxR)
    )
    """
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
    top = []
    if circles is not None:
        top = filter_circles(circles, box)
    overlay = _draw_circles(inpainted_polygon, top, (0,255,0))
    return top, overlay

# -------- 2) Inpaint เอาเม็ดชั้นบนออก --------
def inpaint_top(gray: np.ndarray,
    circles: List[Tuple[float,float,float]],
    expand: float = 1.05) -> np.ndarray:
    """
    สร้าง inpaint mask จากวงกลมชั้นบนแล้วลบออก เพื่อเผยรายละเอียดชั้นล่าง
    """
    mask = np.zeros_like(gray, np.uint8)
    for (x,y,r) in circles:
        cv2.circle(mask, (int(x),int(y)), int(r*expand), 255, -1)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

# -------- 3) ตรวจเม็ดชั้นล่างบนภาพที่ inpaint แล้ว --------
def detect_bottom_layer_from_inpaint(img_bgr: np.ndarray,
        inpainted_gray: np.ndarray,
        box: np.ndarray,
        r_hint_px: Optional[float] = None,
        minR: Optional[float] = None,
        maxR: Optional[float] = None,
        param1: int = 95,
        param2: int = 18,
        min_dist_factor: float = 0.85) -> Tuple[List[Tuple[float,float,float]], np.ndarray]:
    """
    ใช้ HoughCircles แบบไวขึ้น (param2 ต่ำกว่า) บนภาพเทาที่ inpaint แล้ว เพื่อจับเม็ดชั้นล่าง
    """
    
    g = preprocess_img(inpainted_gray, cliLimit=2, titleGridSize=(4,4))
    
    cv2.imwrite("./gray_bottomlayer.png", g)
    
    if r_hint_px is None:
        r_hint_px = max(6, min(img_bgr.shape[:2]) / 60.0)

    # minR = max(4, int(r_hint_px*0.55))
    # maxR = int(r_hint_px*2.45)
    minDist = int(max(4, r_hint_px*min_dist_factor))

    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
        param1=param1, param2=param2,
        minRadius=int(minR), maxRadius=int(maxR)
    )
    
    bottom = []
    if circles is not None:
        bottom = filter_circles(circles, box)
    overlay = _draw_circles(img_bgr, bottom, (255,0,0))
    return bottom, overlay

def inpaint_top_polygon(gray: np.ndarray, 
        box: List[np.ndarray], 
        expand: float = 1.05) -> np.ndarray:
    """
    สร้าง inpaint mask จากพิกัด 4 จุดมุม (Polygon/Rotated Rect) แล้วลบออก
    
    Args:
        gray: ภาพ grayscale input
        boxes: List ของ numpy array ขนาด (4, 2) เก็บพิกัดจุดมุม 4 จุด
        expand: ตัวคูณขยายขนาด (1.05 = ขยาย 5%)
    """
    mask = np.zeros_like(gray, np.uint8)
    
    # box ควรเป็น numpy array float เพื่อการคำนวณที่แม่นยำ
    pts = np.array(box, dtype=np.float32)

    # 1. หาจุดกึ่งกลาง (Centroid) ของสี่เหลี่ยม
    center = np.mean(pts, axis=0)

    # 2. ขยายจุดทั้ง 4 ออกจากจุดกึ่งกลาง
    # สูตร: จุดใหม่ = จุดกึ่งกลาง + (ระยะห่างจากกลาง * ตัวคูณ)
    vectors = pts - center
    expanded_pts = center + (vectors * expand)

    # 3. แปลงกลับเป็น int เพื่อวาดลง Mask
    final_pts = expanded_pts.astype(np.int32)
    
    # วาด Polygon สีขาวลงบน Mask
    # [final_pts] ต้องใส่ใน list เพราะ fillPoly รับ array ของ polygons
    cv2.fillPoly(mask, [final_pts], 255)

    # Inpaint
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

# -------- ฟังก์ชันหลัก: วัดเม็ดทั้งบน-ล่างจากภาพเดียว --------
def measure_beads_with_unpeel(
    img_bgr: np.ndarray,
    box: np.ndarray,
    pixel_mm: float = None,
    r_hint_px: Optional[float] = None,
    calibrate_scale: float = 0.0,
    dedup_center_dist_frac: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    1) จับชั้นบน (Hough เข้มงวด) → circles_top
    2) Inpaint เอาชั้นบนออก → gray_inpaint
    3) จับชั้นล่าง (Hough ไว)  → circles_bottom
    4) รวมผล + กันซ้ำ → DataFrame + overlays
    """
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = inpaint_top_polygon(gray, box, expand=1)
    
    # gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(gray)
    merged = {"upper_size": [], "lower_size": []}
    ov_img = img_bgr.copy()
    gray_inp = gray.copy()
    
    for bead in ['lower_size', 'upper_size']:
        if bead == 'lower_size':
            minR = round((pixel_mm * 0.3) / 2)
            maxR = round((pixel_mm * 0.7) / 2)
            top_param2 = 30
            bottom_param2 = 27
            color_top = (0,0,255)
            color_bot = (255,255,0)
            param1 = 80
        else:
            minR = round((pixel_mm * 0.7) / 2) + 1
            maxR = round((pixel_mm * 1.5) / 2)
            top_param2 = 35
            bottom_param2 = 30
            color_top = (0,255,0)
            color_bot = (255,0,0)
            param1 = 90
            
        # 1) Top
        c_top, ov_top = detect_top_layer(gray_inp, box=box, r_hint_px=r_hint_px, minR=minR, maxR=maxR, param1=param1, param2=top_param2)

        # 2) Inpaint
        gray_inp = inpaint_top(gray_inp, c_top, expand=1.05)

        # 3) Bottom
        c_bot, ov_bot = detect_bottom_layer_from_inpaint(img_bgr, 
                            inpainted_gray=gray_inp,
                            box=box,
                            r_hint_px=r_hint_px,
                            minR=minR, maxR=maxR,
                            param1=param1,
                            param2=bottom_param2)

        ov_img = _draw_circles(ov_img, c_top, color=color_top)
        ov_img = _draw_circles(ov_img, c_bot, color=color_bot)
        # 4) Merge + NMS (กันซ้ำ)
        merged[bead] = _nms_merge_circles(c_bot, c_top, min_center_dist_frac=dedup_center_dist_frac)
    
    merged_all = _nms_merge_circles(merged['upper_size'], merged['lower_size'], min_center_dist_frac=dedup_center_dist_frac)
    # ov_img = _draw_circles(ov_img, merged_all, 	color=(255, 0, 0))
    df_merged = _circles_to_df(merged_all, depth=0, pixel_size_um=pixel_mm, calibrate_scale=calibrate_scale)
    
    size_conv = np.array(df_merged["equiv_diam_umr2"])
    unique_vals, counts = np.unique(size_conv, return_counts=True)
    percentages = (counts / len(size_conv)) * 100
    print("\n")
    for val, pct in zip(unique_vals, percentages):
        print(f"  {val:.2f} mm: {pct:.1f}%")
    print("\n")
    return df_merged, ov_img
    
def measure_beads_with_unpeel_test(
    img_bgr: np.ndarray,
    box: np.ndarray,
    set_param1: Dict["upper_size": int,
                    "lower_size": int], 
    set_param2: Dict["upper_size": Dict["top": int, "bot": int],
                    "lower_size": Dict["top": int, "bot": int]],
    pixel_mm: float = None,
    r_hint_px: Optional[float] = None,
    calibrate_scale: float = 0.0,
    dedup_center_dist_frac: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    1) จับชั้นบน (Hough เข้มงวด) → circles_top
    2) Inpaint เอาชั้นบนออก → gray_inpaint
    3) จับชั้นล่าง (Hough ไว)  → circles_bottom
    4) รวมผล + กันซ้ำ → DataFrame + overlays
    """
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = inpaint_top_polygon(gray, box, expand=1)
    
    # gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(gray)
    merged = {"upper_size": [], "lower_size": []}
    ov_img = img_bgr.copy()
    gray_inp = gray.copy()
    for bead in ['lower_size', 'upper_size']:
        if bead == 'lower_size':
            minR = round((pixel_mm * 0.3) / 2)
            maxR = round(((pixel_mm * 0.7) / 2))
            top_param2 = set_param2["lower_size"]["top"]
            bottom_param2 = set_param2["lower_size"]["bot"]
            color_top = (0,0,255)
            color_bot = (255,255,0)
            param1 = set_param1["lower_size"]
        else:
            minR = round((pixel_mm * 0.7) / 2) + 1
            maxR = round((pixel_mm * 1.5) / 2)
            top_param2 = set_param2["upper_size"]["top"]
            bottom_param2 = set_param2["upper_size"]["bot"]
            color_top = (0,255,0)
            color_bot = (255,0,0)
            param1 = set_param1["upper_size"]
            
        # 1) Top
        c_top, ov_top = detect_top_layer(gray_inp, box=box, r_hint_px=r_hint_px, minR=minR, maxR=maxR, param1=param1, param2=top_param2)

        # 2) Inpaint
        gray_inp = inpaint_top(gray_inp, c_top, expand=1.05)

        # 3) Bottom
        c_bot, ov_bot = detect_bottom_layer_from_inpaint(img_bgr, 
                            inpainted_gray=gray_inp,
                            box=box,
                            r_hint_px=r_hint_px,
                            minR=minR, maxR=maxR,
                            param1=param1,
                            param2=bottom_param2)

        ov_img = _draw_circles(ov_img, c_top, color=color_top)
        ov_img = _draw_circles(ov_img, c_bot, color=color_bot)
        # 4) Merge + NMS (กันซ้ำ)
        merged[bead] = _nms_merge_circles(c_bot, c_top, min_center_dist_frac=dedup_center_dist_frac)
    
    merged_all = _nms_merge_circles(merged['upper_size'], merged['lower_size'], min_center_dist_frac=dedup_center_dist_frac)
    # ov_img = _draw_circles(ov_img, merged_all, 	color=(255, 0, 0))
    df_merged = _circles_to_df(merged_all, depth=0, pixel_size_um=pixel_mm, calibrate_scale=calibrate_scale)

    # cv2.imwrite("./ov_img.png", ov_img)
    return df_merged, ov_img

if __name__ == '__main__':
    import os
    from detect_rectangle import detect_red_rectangles
    
    # --- ฟังก์ชัน Grid Search (คงเดิม) ---
    def grid_search(img, set_param1: Dict, set_param2: Dict, dataset=None, actual_size=None, is_flash=None):
        frame_detected, mean_size, box = detect_red_rectangles(image=img)
        
        # ป้องกัน error กรณีไม่เจอสี่เหลี่ยมเลย
        if mean_size is None or mean_size == 0:
            return dataset if dataset is not None else pd.DataFrame()

        pixel_mm = round(mean_size / 5)
        results = []  # เก็บผลลัพธ์แต่ละรอบ

        # วนลูปพารามิเตอร์
        for lower_param1 in set_param1["lower_size"]:
            for lower_top_param2 in set_param2["lower_size"]["top"]:
                for lower_bot_param2 in set_param2["lower_size"]["bot"]:
                    for upper_param1 in set_param1["upper_size"]:
                        for upper_top_param2 in set_param2["upper_size"]["top"]:
                            for upper_bot_param2 in set_param2["upper_size"]["bot"]:

                                current_param1 = {
                                    "lower_size": lower_param1, "upper_size": upper_param1
                                }
                                current_param2 = {
                                    "lower_size": {"top": lower_top_param2, "bot": lower_bot_param2},
                                    "upper_size": {"top": upper_top_param2, "bot": upper_bot_param2}
                                }

                                # เรียกฟังก์ชันวัดผล (สมมติว่า import มาแล้ว หรือมีอยู่จริง)
                                # *Note: ต้องแน่ใจว่า measure_beads_with_unpeel_test ถูก import หรือ define ไว้แล้ว
                                try:
                                    df, ovs = measure_beads_with_unpeel_test(
                                        img, set_param1=current_param1, set_param2=current_param2,
                                        box=box, pixel_mm=pixel_mm, dedup_center_dist_frac=0.5, r_hint_px=20,
                                    )
                                except NameError:
                                    print("Error: Function 'measure_beads_with_unpeel_test' not found.")
                                    return dataset

                                # วิเคราะห์ผล
                                size_conv = np.array(df['equiv_diam_um']) if not df.empty else []
                                unique_vals, counts = np.unique(size_conv, return_counts=True)
                                percentages = (counts / len(size_conv)) * 100 if len(size_conv) > 0 else []

                                result_row = {
                                    "actual_size": actual_size,
                                    "lower_top_param2": lower_top_param2,
                                    "lower_bottom_param2": lower_bot_param2,
                                    "lower_param1": lower_param1,
                                    "upper_top_param2": upper_top_param2,
                                    "upper_bottom_param2": upper_bot_param2,
                                    "upper_param1": upper_param1,
                                    "is_flash": is_flash,
                                }

                                for uv, pct in zip(unique_vals, percentages):
                                    col_name = f"{uv}"
                                    result_row[col_name] = pct  # เก็บเป็น float เลยเพื่อความง่ายในการคำนวณทีหลัง

                                results.append(result_row)

        results_df = pd.DataFrame(results)
        if dataset is not None:
            dataset = pd.concat([dataset, results_df], ignore_index=True)
        else:
            dataset = results_df

        return dataset

    # --- ส่วนหลักของการทำงาน ---
    root_directory = "dataset"
    all_results_df = pd.DataFrame()

    set_param1 = {
        "upper_size": {80, 90, 100, 110, 120},
        "lower_size": {50, 60, 70}
    }
    set_param2 = {
        "upper_size": {"top": {35, 36, 37}, "bot": {29, 30, 31}},
        "lower_size": {"top": {28, 29, 30, 31}, "bot": {25, 26, 27}}
    }

    print(f"Start processing images in '{root_directory}'...")

    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(root, filename)
                path_parts = os.path.normpath(root).split(os.sep)
                
                # Logic หา is_flash และ actual_size (เหมือนเดิม)
                is_flash = 1 if 'flash_img' in path_parts else (0 if 'not_flash' in path_parts else None)
                actual_size = None
                for part in path_parts:
                    if part.startswith("size"):
                        try:
                            actual_size = float(part.replace("size", ""))
                        except ValueError:
                            continue
                
                if is_flash is None or actual_size is None:
                    continue

                print(f"Processing: {filename} | Size={actual_size} | Flash={is_flash}")
                image = cv2.imread(filepath)
                if image is None: continue

                all_results_df = grid_search(
                    img=image, set_param1=set_param1, set_param2=set_param2,
                    dataset=all_results_df, actual_size=actual_size, is_flash=is_flash
                )

    # -------------------------------------------------
    # 🎯 ส่วนที่เพิ่มเติม: สรุปผลและวิเคราะห์ค่าเฉลี่ย
    # -------------------------------------------------
    output_path = "grid_search_results_all.csv"
    summary_path = "grid_search_summary_ranking.csv"

    if not all_results_df.empty:
        # 1. Fill NaN with 0 (กรณีไซส์ไหนไม่เจอก็คือ 0%)
        all_results_df = all_results_df.fillna(0)

        # 2. ฟังก์ชันดึงค่า % ความแม่นยำของ actual_size ใน row นั้นๆ
        def get_target_accuracy(row):
            # แปลง actual_size เป็น string เพื่อหาชื่อคอลัมน์ (เช่น 0.7 -> "0.7")
            target_col = str(row['actual_size'])
            
            # บางครั้ง pandas อาจมองชื่อคอลัมน์เป็น float หรือ string ต้องลองเช็ค
            if target_col in row:
                return float(row[target_col])
            
            # กรณีชื่อคอลัมน์อาจจะเป็น float ใน dataframe (เช่น 0.7)
            if row['actual_size'] in row:
                return float(row[row['actual_size']])
                
            return 0.0

        # สร้างคอลัมน์ใหม่ 'accuracy_score' คือ % ที่ทายถูกตาม actual_size
        all_results_df['accuracy_score'] = all_results_df.apply(get_target_accuracy, axis=1)

        # 3. จัดกลุ่ม (Group By) พารามิเตอร์ เพื่อหาค่าเฉลี่ยความแม่นยำ
        group_cols = [
            "lower_param1", "upper_param1",
            "lower_top_param2", "lower_bottom_param2",
            "upper_top_param2", "upper_bottom_param2",
            "is_flash"
        ]

        # คำนวณค่าเฉลี่ยของ accuracy_score
        summary_df = all_results_df.groupby(group_cols)['accuracy_score'].mean().reset_index()
        
        # เรียงลำดับจากแม่นยำมากที่สุด -> น้อยที่สุด
        summary_df = summary_df.sort_values(by='accuracy_score', ascending=False)

        # 4. บันทึกไฟล์
        # ไฟล์ Raw data ทั้งหมด
        all_results_df.to_csv(output_path, index=False)
        
        # ไฟล์สรุป Ranking พารามิเตอร์ที่ดีที่สุด
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"✅ Process Complete!")
        print(f"1. Raw Results saved to: {output_path}")
        print(f"2. Summary Ranking saved to: {summary_path}")
        print(f"\n🏆 Top 3 Best Parameters (Highest Average Accuracy):")
        print(summary_df.head(3).to_string())
        print(f"{'='*60}")

    else:
        print("No results generated.")