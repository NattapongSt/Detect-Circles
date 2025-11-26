import cv2
import numpy as np

def detect_red_rectangles(image=None, image_path=None, min_area=1000):
    if image is None and image_path:
        image = cv2.imread(image_path)
    
    image = cv2.medianBlur(image, 3)  # ลด noise เล็กน้อย
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)            # แปลงเป็น HSV เพราะการแยกสีใน HSV ทำได้ง่ายกว่า
    
    """
    สีแดงใน HSV จะอยู่ สองช่วง (Hue ใกล้ 0° และ 180°) 
    np.array([H, S, V])
    
    H (Hue) = โทนสี 0–180
    S (Saturation) = ความสดของสี 0–255
    V (Value/Brightness) = ความสว่าง 0–255
    """
    lower_red1 = np.array([0, 110, 85])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 110, 85])
    upper_red2 = np.array([180, 255, 255])

    # mask (ขาว=ใช่, ดำ=ไม่ใช่)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # ทำความสะอาด noise เล็กน้อย
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)

    # หาเส้นรอบวัตถุ (contours) จาก mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size = []
    box = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # สร้างกรอบสี่เหลี่ยมที่หมุนได้
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue

        # แปลงเป็นพิกัดมุม 4 จุด
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        if box is None or len(box) == 0:
            continue

        cv2.drawContours(image, [box], 0, (0, 255, 0), 1)

        size = [w, h]

    if size:
        mean_size = np.mean(size)
    else:
        mean_size = 0
        
    return image,  mean_size, box
    
if __name__ == "__main__":
    img, mean_size, box = detect_red_rectangles(image_path="/home/diea/IRPC_Internship68/EPS_Detection/capture_images/capture_1758860802.jpg")
    cv2.imwrite("Detected_Red_Rectangles.png", img)