# cam_control.py — camera thread + optional preview window + vision helper
import cv2
import threading
import time
import base64
import requests

# --- Ollama vision settings ---
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
VISION_MODEL = "llava:7b"


def describe_image_llava(
    frame_bgr,
    prompt=(
        "Describe what you see in at most 4 short sentences. "
        "Be concise. The entire answer must be under 60 words. "
        "If there is visible text, add one last sentence starting with 'Text: ' and write the text."
)


        ):
    
    """
    Take an OpenCV BGR frame, send it to Ollama LLaVA model, and
    return the textual description.
    """
    # 1) Encode frame as JPEG
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return "Failed to encode camera frame."

    # 2) JPEG -> base64 string
    b64 = base64.b64encode(buf).decode("ascii")

    # 3) Build Ollama request payload (vision)
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "stream": False,
        "images": [b64],
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip() or "Model returned empty response."
    except Exception as e:
        return f"Vision request failed: {e}"


class CameraControl:
    def __init__(self, index=0, width=1280, height=720, preview=False, window_name="RiadCam"):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.last_frame = None

        # preview
        self.preview = preview
        self.window_name = window_name
        self._preview_running = False
        self._preview_thread = None

    @property
    def is_open(self):
        return self.running

    def open(self):
        if self.running:
            return True
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            if self.cap:
                self.cap.release()
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

        if self.preview:
            self.show_preview()
        return True

    def _loop(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.last_frame = frame
        if self.cap:
            self.cap.release()
            self.cap = None

    def close(self):
        self.running = False
        self.hide_preview()

    def get_frame(self):
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    # ------- high-level vision helper -------
    def describe_current_view(self, prompt="Shortly describe what is in front of the camera."):
        """
        Grab the latest frame and ask LLaVA to describe it.
        """
        frame = self.get_frame()
        if frame is None:
            return "Camera frame is not available yet."
        return describe_image_llava(frame, prompt)

    # ------- preview controls -------
    def show_preview(self):
        if self._preview_running:
            return
        self.preview = True
        self._preview_running = True
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def hide_preview(self):
        self.preview = False
        self._preview_running = False
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def _preview_loop(self):
        # non-blocking imshow loop
        while self._preview_running:
            frame = self.get_frame()
            if frame is not None:
                cv2.imshow(self.window_name, frame)
                # ESC closes the preview window, camera keeps running
                if (cv2.waitKey(1) & 0xFF) == 27:
                    self.hide_preview()
                    break
            else:
                # if no frame yet
                cv2.waitKey(1)
            time.sleep(0.01)
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
