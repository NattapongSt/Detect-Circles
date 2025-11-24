import cv2, numpy as np, pandas as pd
from typing import List, Tuple, Optional, Dict
import cv2

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
            row["radius_um"]      = round(row["radius_px"] / pixel_size_um, 1)
            row["equiv_diam_um"]  = round(row["equiv_diam_px"] / pixel_size_um, 1)
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

# -------- 1) ตรวจเม็ดชั้นบน (คม/สว่าง) --------
def detect_top_layer(img_bgr: np.ndarray,
                     r_hint_px: Optional[float] = None,
                     minR: Optional[float] = None,
                     maxR: Optional[float] = None,
                     param2: int = 22,
                     min_dist_factor: float = 0.9) -> Tuple[List[Tuple[float,float,float]], np.ndarray]:
    """
    ใช้ HoughCircles ที่เข้มงวดกว่าปกติ เพื่อจับเม็ด 'ชั้นบน' ที่ขอบคม-สว่าง
    return circles (x,y,r) และ overlay ของรอบนี้
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)

    if r_hint_px is None:
        r_hint_px = max(6, min(img_bgr.shape[:2]) / 60.0)

    # minR = max(4, int(r_hint_px*0.6))
    # maxR = int(r_hint_px*2.5)
    minDist = int(max(4, r_hint_px*min_dist_factor))

    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
        param1=120, param2=param2,
        minRadius=int(minR), maxRadius=int(maxR)
    )
    top = []
    if circles is not None:
        top = [tuple(map(float, c)) for c in np.around(circles[0,:]).astype(np.float32)]
    overlay = _draw_circles(img_bgr, top, (0,255,0))
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
                                     r_hint_px: Optional[float] = None,
                                     minR: Optional[float] = None,
                                     maxR: Optional[float] = None,
                                     param2: int = 18,
                                     min_dist_factor: float = 0.85) -> Tuple[List[Tuple[float,float,float]], np.ndarray]:
    """
    ใช้ HoughCircles แบบไวขึ้น (param2 ต่ำกว่า) บนภาพเทาที่ inpaint แล้ว เพื่อจับเม็ดชั้นล่าง
    """
    g = cv2.medianBlur(inpainted_gray, 3)
    if r_hint_px is None:
        r_hint_px = max(6, min(img_bgr.shape[:2]) / 60.0)

    # minR = max(4, int(r_hint_px*0.55))
    # maxR = int(r_hint_px*2.45)
    minDist = int(max(4, r_hint_px*min_dist_factor))

    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
        param1=120, param2=param2,
        minRadius=int(minR), maxRadius=int(maxR)
    )
    bottom = []
    if circles is not None:
        bottom = [tuple(map(float, c)) for c in np.around(circles[0,:]).astype(np.float32)]
    overlay = _draw_circles(img_bgr, bottom, (255,0,0))
    return bottom, overlay

# -------- ฟังก์ชันหลัก: วัดเม็ดทั้งบน-ล่างจากภาพเดียว --------
def measure_beads_with_unpeel(
    img_bgr: np.ndarray,
    pixel_mm: float = None,
    r_hint_px: Optional[float] = None,
    dedup_center_dist_frac: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    1) จับชั้นบน (Hough เข้มงวด) → circles_top
    2) Inpaint เอาชั้นบนออก → gray_inpaint
    3) จับชั้นล่าง (Hough ไว)  → circles_bottom
    4) รวมผล + กันซ้ำ → DataFrame + overlays
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    merged = {"upper_size": [], "lower_size": []}
    ov_img = img_bgr.copy()
    
    for bead in ['lower_size', 'upper_size']:
         
        if bead == 'lower_size':
            minR = (pixel_mm * 0.3) / 2
            maxR = (pixel_mm * 0.6) / 2
            top_param2 = 26
            bottom_param2 = 21
        else:
            minR = (pixel_mm * 0.7) / 2
            maxR = (pixel_mm * 1.5) / 2
            top_param2 = 35.8
            bottom_param2 = 31.8
         
        # 1) Top
        c_top, ov_top = detect_top_layer(img_bgr, r_hint_px=r_hint_px, minR=minR, maxR=maxR, param2=top_param2)

        # 2) Inpaint
        gray_inp = inpaint_top(gray, c_top, expand=1.05)

        # 3) Bottom
        c_bot, ov_bot = detect_bottom_layer_from_inpaint(img_bgr, gray_inp,
                                                        r_hint_px=r_hint_px,
                                                        minR=minR, maxR=maxR,
                                                        param2=bottom_param2)

        # 4) Merge + NMS (กันซ้ำ)
        merged[bead] = _nms_merge_circles(c_top, c_bot, min_center_dist_frac=dedup_center_dist_frac)
    
    merged_all = _nms_merge_circles(merged['lower_size'], merged['upper_size'], min_center_dist_frac=dedup_center_dist_frac)
    ov_img = _draw_circles(ov_img, merged_all, 	color=(255, 0, 0))
    df_merged = _circles_to_df(merged_all, depth=0, pixel_size_um=pixel_mm)

    return df_merged, ov_img

if __name__ == '__main__':
    img = cv2.imread("/home/diea/IRPC_Internship68/EPS_Detection/capture_images/capture_1758867863.jpg")  # BGR
    df, ovs = measure_beads_with_unpeel(
        img,
        pixel_size_um=None,      # ใส่สเกลจริงถ้ามี เช่น 2.5
        r_hint_px=20,          # ถ้ารู้คร่าว ๆ ใส่ได้ เช่น 10
        top_param2=31,           # ชั้นบนเอาค่าเข้มงวด (ลด false positive)
        bottom_param2=26         # ชั้นล่างให้ไวขึ้น (จับขอบจาง)
    )

    cv2.imwrite("./overlay_images/overlay_top.png", ovs['all'])