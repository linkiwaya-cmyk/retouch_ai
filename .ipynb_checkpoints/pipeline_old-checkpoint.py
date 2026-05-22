cat > pipeline.py << 'ENDOFFILE'
import requests
import json
import time
import io
import os
from PIL import Image
import pillow_heif

RETOUCH_TOKEN = os.getenv("RETOUCH4ME_TOKEN")
BASE_URL = "https://cf-retoucher.retouch4.me/api/v1"

PRESET = {
    "mode": "professional",
    "tasks": [
        {"Plugin": "Heal",             "Scale": 0, "Alpha1": 1.0},
        {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.39},
        {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 1.0},
        {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.5},
        {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.25, "Alpha2": 0.25},
        {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 1.0,  "Alpha2": 0.2},
        {"Plugin": "Skin Tone",        "Scale": 0, "Alpha1": 1.0,  "Alpha2": 1.0},
        {"Plugin": "Portrait Volumes", "Scale": 0, "Alpha1": 0.5}
    ]
}

def _to_jpeg(image_bytes: bytes, filename: str) -> tuple:
    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    new_name = filename.rsplit('.', 1)[0] + '.jpg'
    return buf.getvalue(), new_name

def process_image(image_bytes: bytes, filename: str) -> bytes:
    jpeg_bytes, jpeg_name = _to_jpeg(image_bytes, filename)

    resp = requests.post(
        f"{BASE_URL}/retoucher/start",
        files={"file": (jpeg_name, jpeg_bytes, "image/jpeg")},
        data={"token": RETOUCH_TOKEN, "payload": json.dumps(PRESET)},
        timeout=30
    )
    data = resp.json()
    if data.get("status") != 200:
        raise Exception(f"Retouch4me error: {data}")

    task_id = data["id"]

    for _ in range(90):
        time.sleep(2)
        s = requests.get(
            f"{BASE_URL}/retoucher/status/{task_id}",
            timeout=10
        ).json()
        if s.get("state") == "completed":
            break
        if s.get("state") == "failed":
            raise Exception(f"Retouch4me failed: {s.get('reason')}")
    else:
        raise Exception("Timeout")

    result = requests.get(
        f"{BASE_URL}/retoucher/getFile/{task_id}",
        timeout=30
    )
    return result.content

def get_balance() -> int:
    try:
        r = requests.post(
            "https://3dlutcreator.com/api/retouch/v1/balance",
            data={"retouchtoken": RETOUCH_TOKEN, "modes[]": "professional"},
            timeout=10
        )
        return r.json().get("remaining", {}).get("professional", 0)
    except:
        return -1
ENDOFFILE