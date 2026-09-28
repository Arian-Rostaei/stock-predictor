import webview
import subprocess
import time
import sys
import os
import signal
import atexit

# مسیر پروژه
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_FILE = os.path.join(PROJECT_DIR, "dashboard.py")

# متغیر برای نگه‌داشتن پروسه Streamlit
streamlit_process = None

def cleanup():
    """بستن Streamlit وقتی اپ بسته می‌شه"""
    global streamlit_process
    if streamlit_process:
        print("Shutting down Streamlit...")
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()

# ثبت تابع cleanup برای اجرا در هنگام خروج
atexit.register(cleanup)

def start_streamlit():
    """اجرای Streamlit در پس‌زمینه"""
    global streamlit_process
    print("Starting Streamlit server...")
    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", DASHBOARD_FILE,
            "--server.headless=true",
            "--server.port=8501",
            "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none"  # خاموش کردن file watcher برای مصرف کمتر رم
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=PROJECT_DIR
    )

def wait_for_streamlit(timeout=30):
    """منتظر می‌مونه تا Streamlit آماده بشه"""
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen("http://localhost:8501", timeout=2)
            print("Streamlit is ready!")
            return True
        except Exception:
            time.sleep(1)
    return False

if __name__ == "__main__":
    # اجرای Streamlit
    start_streamlit()
    
    # منتظر آماده شدنش بمون
    if not wait_for_streamlit():
        print("Error: Streamlit failed to start in time.")
        cleanup()
        sys.exit(1)
    
    # ساخت پنجره دسکتاپ
    print("Opening desktop window...")
    window = webview.create_window(
        title="📈 پیش‌بینی بازار سهام",
        url="http://localhost:8501",
        width=1300,
        height=850,
        resizable=True,
        min_size=(900, 600)
    )
    
    # شروع حلقه اصلی Webview
    webview.start()
    
    # بعد از بستن پنجره، Streamlit رو ببند
    cleanup()
