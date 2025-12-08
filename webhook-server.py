from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
import subprocess
import os

app = FastAPI()

# ตั้งค่า Secret Token
GITLAB_SECRET_TOKEN = "EPS-Beads-Size-Meter-6516"

# ฟังก์ชันที่จะให้ทำงานลับหลัง (Background)
def run_update_script():
    try:
        print("Starting background update process...")
        # รัน Shell Script (ตรวจสอบ path ให้ถูกต้อง)
        subprocess.run(['/home/eps/eps-bead-size-meter/auto-update-code.sh'], shell=True, check=True)
        print("Update process finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during update: {e}")

@app.post("/webhook")
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_token: str = Header(None) # รับ Header อัตโนมัติ
):
    # 1. ตรวจสอบ Token
    if x_gitlab_token != GITLAB_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    # 2. ตรวจสอบ Event Type
    event_type = request.headers.get('X-Gitlab-Event')
    if event_type == "Push Hook":
        # 3. สั่งให้รัน Update ใน Background (ตอบกลับ GitLab ทันที ไม่ต้องรอ script เสร็จ)
        background_tasks.add_task(run_update_script)
        return {"message": "Update started in background"}

    return {"message": "Event ignored"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)