import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
from pathlib import Path
import random
import platform
import time
import datetime
import threading
import collections
import json
from phone_stream import MJPEGStreamer, make_qr_image

import sys as _sys, os as _os
# 后台隐藏运行时（pythonw 无控制台）把输出写入日志文件，便于排查问题
if (not (_sys.stdout and getattr(_sys.stdout, "isatty", lambda: False)()) and
        _os.environ.get("YOLO_FORCE_STDOUT", "0").lower() not in {"1", "true", "yes", "on"}):
    try:
        _log_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "yolo_cam.log")
        _logf = open(_log_path, "a", encoding="utf-8", buffering=1)
        _sys.stdout = _logf
        _sys.stderr = _logf
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()

# ====== Person detector selection ======
# Only accept a detector whose metadata explicitly contains the COCO "person"
# class. A custom checkpoint with names={0: "item"} previously produced
# focus_id=-1 and made classes=[-1] suppress every detection.
_MODEL_CANDIDATES = [
    "models/yolov8n.pt", "yolov8n.pt",
    "models/yolov8m.pt", "yolov8m.pt",
    "models/yolov8n_hard.pt", "yolov8n_hard.pt",
]


def _find_person_class(names):
    items = names.items() if isinstance(names, dict) else enumerate(names or [])
    for cid, name in items:
        if str(name).strip().lower() == "person":
            return int(cid)
    return None


model_path = None
model = None
for _cand in _MODEL_CANDIDATES:
    _p = BASE_DIR / _cand
    if not _p.exists():
        continue
    try:
        _candidate_model = YOLO(str(_p))
        if _find_person_class(_candidate_model.names) is None:
            print(f"Skipping non-person detector: {_p} names={_candidate_model.names}")
            continue
        model_path = _p
        model = _candidate_model
        break
    except Exception as _model_exc:
        print(f"Skipping broken detector {_p}: {_model_exc}")

if model is None:
    model_path = BASE_DIR / "models" / "yolov8n.pt"
    print(f"No local person detector found, loading: {model_path}")
    model = YOLO(str(model_path))
    if _find_person_class(model.names) is None:
        raise RuntimeError(f"Person class missing from detector: {model.names}")

print(f"Loading detection model: {model_path} (person class ready)")
POSE_MODEL_PATH = BASE_DIR / "models" / "yolov8n-pose.pt"
POSE_MODEL_NAME = "yolov8n-pose.pt"
pose_model = None
pose_model_ready = False
pose_model_loading = False
pose_load_error = None
pose_load_lock = threading.Lock()
pose_people_count = 0
pose_visible_keypoints = 0
detected_people_count = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True   # 输入尺寸固定时显著加速推理
print(f"Device: {device}")
print(f"OS: {platform.system()} {platform.release()}")

# ====== Font ======
font_path = None
_font_cache = {}

# Prefer Microsoft YaHei for cleaner Chinese UI; fall back to common Windows CJK fonts.
font_candidates = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Deng.ttf",
    "C:/Windows/Fonts/Dengb.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/simsunb.ttf",
]

for fp in font_candidates:
    if Path(fp).exists():
        font_path = fp
        print(f"Font: {fp}")
        break

if not font_path:
    print("Warning: No Chinese font found, using default")


def get_font(size):
    key = (font_path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    return _font_cache[key]


def text_pixel_width(text, size):
    bbox = get_font(size).getbbox(str(text))
    return max(0, bbox[2] - bbox[0])


def ellipsize_text(text, max_width, size):
    text = str(text)
    if max_width <= 0:
        return ""
    if text_pixel_width(text, size) <= max_width:
        return text
    suffix = "..."
    if text_pixel_width(suffix, size) > max_width:
        return ""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if text_pixel_width(text[:mid] + suffix, size) <= max_width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + suffix


def draw_cn(img, text, pos, size=18, color=(0, 255, 0), bg=None, anchor="lt"):
    """Draw Chinese text by converting only its small ROI instead of the full frame."""
    text = str(text)
    if not text:
        return img
    font = get_font(size)
    probe = Image.new("RGB", (1, 1))
    d_probe = ImageDraw.Draw(probe)
    bbox = d_probe.textbbox((0, 0), text, font=font)
    x, y = map(float, pos)
    if anchor == "mm":
        x -= (bbox[0] + bbox[2]) / 2.0
        y -= (bbox[1] + bbox[3]) / 2.0
    elif anchor == "rt":
        x -= bbox[2]
        y -= bbox[1]

    pad_x = 10 if bg else 3
    pad_y = 7 if bg else 3
    left = int(np.floor(x + bbox[0] - pad_x))
    top = int(np.floor(y + bbox[1] - pad_y))
    right = int(np.ceil(x + bbox[2] + pad_x))
    bottom = int(np.ceil(y + bbox[3] + pad_y))
    h, w = img.shape[:2]
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(w, right), min(h, bottom)
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    lx, ly = x - x1, y - y1
    if bg:
        d.rounded_rectangle(
            [lx + bbox[0] - 8, ly + bbox[1] - 5,
             lx + bbox[2] + 8, ly + bbox[3] + 7],
            radius=8, fill=bg)
    d.text((lx, ly), text, font=font, fill=color)
    img[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def render_text_patch(text, size, color):
    """Render a reusable transparent text patch for static button labels."""
    text = str(text)
    font = get_font(size)
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    patch = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    draw.text((1 - bbox[0], 1 - bbox[1]), text, font=font,
              fill=tuple(color) + (255,))
    rgba = np.asarray(patch)
    return rgba[:, :, :3][:, :, ::-1].copy(), rgba[:, :, 3].copy()


def blit_text_patch(img, patch_bgr, alpha, center):
    """Alpha-blend a cached text patch centered at the requested position."""
    ph, pw = alpha.shape[:2]
    cx, cy = map(int, center)
    x1, y1 = cx - pw // 2, cy - ph // 2
    x2, y2 = x1 + pw, y1 + ph
    ih, iw = img.shape[:2]
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(iw, x2), min(ih, y2)
    if dx2 <= dx1 or dy2 <= dy1:
        return img
    sx1, sy1 = dx1 - x1, dy1 - y1
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
    a = alpha[sy1:sy2, sx1:sx2, None].astype(np.float32) / 255.0
    src = patch_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    dst = img[dy1:dy2, dx1:dx2].astype(np.float32)
    img[dy1:dy2, dx1:dx2] = (src * a + dst * (1.0 - a)).astype(np.uint8)
    return img


# ====== COCO Chinese ======
COCO_CN = {
    "person": "人", "bicycle": "自行车", "car": "汽车",
    "motorcycle": "摩托车", "airplane": "飞机", "bus": "公交车",
    "train": "火车", "truck": "卡车", "boat": "船",
    "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停止标志", "parking meter": "停车计时器",
    "bench": "长椅", "bird": "鸟", "cat": "猫", "dog": "狗",
    "horse": "马", "sheep": "羊", "cow": "牛", "elephant": "大象",
    "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿",
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包",
    "tie": "领带", "suitcase": "行李箱", "frisbee": "飞盘",
    "skis": "滑雪板", "snowboard": "单板滑雪", "sports ball": "运动球",
    "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板",
    "surfboard": "冲浪板", "tennis racket": "网球拍",
    "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子",
    "fork": "叉子", "knife": "刀", "spoon": "勺子", "bowl": "碗",
    "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈",
    "cake": "蛋糕", "chair": "椅子", "couch": "沙发",
    "potted plant": "盆栽", "bed": "床", "dining table": "餐桌",
    "toilet": "马桶", "tv": "电视", "laptop": "笔记本",
    "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘",
    "cell phone": "手机", "microwave": "微波炉", "oven": "烤箱",
    "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱",
    "book": "书", "clock": "钟", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "泰迪熊", "hair drier": "吹风机", "toothbrush": "牙刷",
}

def get_cn(label):
    if label in COCO_CN:
        return COCO_CN[label]
    for en, cn in COCO_CN.items():
        if en in label.lower():
            return cn
    return label

random.seed(42)
cls_color = {}

def get_color(cid):
    if cid not in cls_color:
        cls_color[cid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return cls_color[cid]

cn_to_en = {cn: en for en, cn in COCO_CN.items()}
all_names = model.names
PERSON_CN = "人"
PERSON_CONF = 0.35   # 降低阈值以提升识别率（更少漏检）
PERSON_IOU = 0.45
POSE_KEYPOINT_CONF = 0.15
COCO_POSE_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6),
)
MIN_PERSON_BOX_AREA = 5000
MIN_PERSON_BOX_HEIGHT = 80

# ====== Snapshot (鎷嶇収) Settings ======
SNAP_DIR = BASE_DIR / "snapshots"
SNAP_DIR.mkdir(exist_ok=True)
MAX_STORAGE_GB = 10  # Max storage size in GB (adjustable)
PLAYBACK_SNAP_LIMIT = int(_os.environ.get("YOLO_PLAYBACK_SNAP_LIMIT", "200"))  # 鎵嬫満鍥炴斁鏈€澶氳繑鍥炵殑鐓х墖鏁?
snapshot_lock = threading.RLock()
_snapshot_start_meta = []
for _p in SNAP_DIR.iterdir():
    if not _p.is_file() or _p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    try:
        _st = _p.stat()
        _snapshot_start_meta.append((_st.st_mtime, _st.st_size, _p))
    except OSError:
        pass
snapshot_count = len(_snapshot_start_meta)
_snapshot_total_bytes = sum(_item[1] for _item in _snapshot_start_meta)
_snapshot_playback_lock = threading.Lock()
_snapshot_playback_items = [
    {"name": _path.name, "type": "image", "url": f"/snap/{_path.name}", "size": _size}
    for _, _size, _path in sorted(_snapshot_start_meta, key=lambda _item: _item[0], reverse=True)[:PLAYBACK_SNAP_LIMIT]
]


def _remember_snapshot_for_playback(path, size):
    item = {"name": path.name, "type": "image", "url": f"/snap/{path.name}", "size": int(size)}
    with _snapshot_playback_lock:
        _snapshot_playback_items.insert(0, item)
        del _snapshot_playback_items[PLAYBACK_SNAP_LIMIT:]

# ====== 褰曞埗锛堝綍灞忓洖鏀撅級 ======
REC_DIR = BASE_DIR / "recordings"
REC_DIR.mkdir(exist_ok=True)
recording = False
video_writer = None
rec_lock = threading.Lock()
REC_FPS = 20

def start_recording():
    global recording, video_writer
    with rec_lock:
        if recording:
            return False
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fname = REC_DIR / f"{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(fname), fourcc, REC_FPS, (OUTPUT_W, OUTPUT_H))
        if not vw.isOpened():
            print("  [录制] 无法打开视频写入器")
            return False
        video_writer = vw
        recording = True
    print(f"  [录制] 开始 -> {fname.name}")
    return True

def stop_recording():
    global recording, video_writer
    with rec_lock:
        recording = False
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
            video_writer = None
    print("  [录制] 已停止")

def toggle_recording():
    if recording:
        stop_recording()
    else:
        start_recording()

def list_playback_files():
    """Return recent recordings and cached snapshots without a full directory scan."""
    items = []
    try:
        for f in sorted(REC_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                items.append({"name": f.name, "type": "video",
                              "url": f"/file/{f.name}", "size": f.stat().st_size})
        with _snapshot_playback_lock:
            items.extend(dict(item) for item in _snapshot_playback_items)
    except Exception:
        pass
    return items

pending_main_actions = collections.deque()


def phone_command(action, idx=None):
    """手机端发来的控制指令。"""
    if action == "snapshot":
        do_manual_snap()
        return "snapshot ok"
    elif action in ("detection_toggle", "monitor_toggle"):
        do_toggle_detection()
        return "detection on" if detection_enabled else "monitor on"
    elif action == "multi_toggle":
        with pending_actions_lock:
            if "multi_toggle" not in pending_main_actions and not camera_switching:
                pending_main_actions.append("multi_toggle")
                return "multi queued"
        return "multi busy"
    elif action == "pose_toggle":
        with pending_actions_lock:
            if "pose_toggle" not in pending_main_actions:
                pending_main_actions.append("pose_toggle")
                return "pose queued"
        return "pose busy"
    elif action == "auto_snap_toggle":
        do_toggle_auto_snap()
        return "auto snap on" if auto_snap else "auto snap off"
    elif action in ("record", "rec_toggle"):
        toggle_recording()
        return "recording on" if recording else "recording off"
    elif action == "rec_stop":
        if recording:
            stop_recording()
        return "recording off"
    elif action == "stream_profile":
        profile = str(idx or "balanced").strip().lower()
        if streamer.set_profile(profile):
            return f"stream profile {profile}"
        return "stream profile invalid"
    elif action == "select_cam":
        # 閫夋憚鍍忓ご锛氳法绾跨▼瀹夊叏鍏ラ槦锛岀敱涓诲惊鐜湪涓荤嚎绋嬫墽琛屽垏鎹€?
        try:
            idx = int(idx) if idx not in (None, "") else -1
        except Exception:
            idx = -1
        if idx < 0 or multi_cam or camera_switching:
            return "select invalid"
        with pending_actions_lock:
            if not any(isinstance(a, tuple) and a[0] == "select_cam" and a[1] == idx
                       for a in pending_main_actions):
                pending_main_actions.append(("select_cam", idx))
        return "select queued"
    return "unknown"

def get_max_storage_bytes():
    return MAX_STORAGE_GB * 1024 * 1024 * 1024

def get_dir_size_mb():
    """Return cached snapshot storage usage without rescanning thousands of files."""
    return _snapshot_total_bytes / (1024 * 1024)


def cleanup_old_snapshots():
    """Delete oldest snapshots only when the cached usage approaches the limit."""
    global _snapshot_total_bytes, snapshot_count
    max_bytes = get_max_storage_bytes()
    target_bytes = int(max_bytes * 0.95)
    if _snapshot_total_bytes <= target_bytes:
        return

    entries = []
    for path in SNAP_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            entries.append((stat.st_mtime, stat.st_size, path))
        except OSError:
            continue
    entries.sort(key=lambda item: item[0])
    # Re-sync once when cleanup is actually needed, then remove oldest first.
    _snapshot_total_bytes = sum(item[1] for item in entries)
    for _, size, oldest in entries:
        if _snapshot_total_bytes <= target_bytes:
            break
        try:
            oldest.unlink()
            _snapshot_total_bytes = max(0, _snapshot_total_bytes - size)
            snapshot_count = max(0, snapshot_count - 1)
            print(f"  [??] ???????: {oldest.name} ({get_dir_size_mb():.1f} MB)")
        except OSError:
            pass

def add_watermark(img):
    """Add a timestamp watermark (down to the second) to the bottom of the image."""
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    h, w = img.shape[:2]
    # 半透明底栏（任何宽度下都有背景，保证文字清晰可读）
    strip_h = 32
    strip_h = min(strip_h, max(18, h // 6))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    # 用 PIL 绘制（支持中文）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    try:
        bbox = font.getbbox(ts)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = len(ts) * 10
    tx = max(8, w - tw - 12)
    ty = h - strip_h + max(3, (strip_h - 18) // 2)
    d.text((tx, ty), ts, font=font, fill=(255, 255, 200))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def _save_snapshot_unlocked(frame, label, conf, full_frame=False):
    """Encode and save one snapshot while the caller holds snapshot_lock."""
    global last_snap_path, snapshot_count, _snapshot_total_bytes
    cleanup_old_snapshots()

    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    safe_label = label.replace("/", "_").replace("\\", "_")

    # 动态调整照片分辨率，避免图片太大占用空间
    max_snap_w = 1280
    max_snap_h = 720
    fh, fw = frame.shape[:2]
    if fw > max_snap_w or fh > max_snap_h:
        scale = min(max_snap_w / fw, max_snap_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if full_frame:
        # Full frame with watermark
        watermarked = add_watermark(frame.copy())
        fname = f"{ts}_全屏_{safe_label}_{conf:.2f}.jpg"
    else:
        # Cropped object (人物抓拍/手动裁剪) 同样带「到秒」时间水印
        watermarked = add_watermark(frame.copy())
        fname = f"{ts}_{safe_label}_{conf:.2f}.jpg"

    fpath = SNAP_DIR / fname

    try:
        # 注意：本环境（opencv 4.13 + 中文路径 D:\yolo识别\...）下 cv2.imwrite 会
        # 静默返回 False、文件写不出；改用 cv2.imencode + 原生 open 写入，Windows 走
        # Unicode API，支持中文目录/文件名，且对 ASCII 路径同样有效。
        ok_enc, buf = cv2.imencode(".jpg", watermarked,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_enc:
            raise RuntimeError("imencode failed")
        jpeg_bytes = buf.tobytes()
        with open(str(fpath), "wb") as _f:
            _f.write(jpeg_bytes)
        _snapshot_total_bytes += len(jpeg_bytes)
        _remember_snapshot_for_playback(fpath, len(jpeg_bytes))
        print(f"  [拍照] {fname}  ({get_dir_size_mb():.1f} MB / {MAX_STORAGE_GB} GB)")
        last_snap_path = str(fpath)
        snapshot_count += 1
        return True
    except Exception as e:
        print(f"  [拍照失败] {e}")
        return False


def save_snapshot(frame, label, conf, full_frame=False):
    """Thread-safe snapshot entry used by desktop, phone and auto capture."""
    with snapshot_lock:
        return _save_snapshot_unlocked(frame, label, conf, full_frame=full_frame)

def get_boxes_data(boxes):
    if boxes is None:
        return [], [], [], []
    if isinstance(boxes, dict):
        return boxes.get('xyxy', boxes.get('boxes', [])), boxes.get('conf', []), boxes.get('cls', []), boxes.get('track_id', [])
    track_ids = []
    if getattr(boxes, 'id', None) is not None:
        track_ids = boxes.id.int().cpu().tolist()
    return boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), track_ids


def empty_inference_cache():
    return {
        'xyxy': np.empty((0, 4)),
        'conf': np.array([]),
        'cls': np.array([]),
        'track_id': np.array([]),
        'keypoints': np.empty((0, 17, 2)),
        'keypoint_conf': np.empty((0, 17)),
    }


def result_to_cache(result):
    cache = empty_inference_cache()
    boxes = getattr(result, 'boxes', None)
    if boxes is not None and len(boxes) > 0:
        cache['xyxy'] = boxes.xyxy.clone().cpu().numpy()
        cache['conf'] = boxes.conf.clone().cpu().numpy()
        cache['cls'] = boxes.cls.clone().cpu().numpy()
        if getattr(boxes, 'id', None) is not None:
            cache['track_id'] = boxes.id.int().cpu().numpy()

    keypoints = getattr(result, 'keypoints', None)
    if keypoints is not None and len(keypoints) > 0:
        cache['keypoints'] = keypoints.xy.clone().cpu().numpy()
        keypoint_conf = getattr(keypoints, 'conf', None)
        if keypoint_conf is not None:
            cache['keypoint_conf'] = keypoint_conf.clone().cpu().numpy()
        else:
            cache['keypoint_conf'] = np.ones(cache['keypoints'].shape[:2], dtype=np.float32)
    return cache


def get_pose_metrics(cache):
    if not isinstance(cache, dict):
        return 0, 0
    keypoints = np.asarray(cache.get('keypoints', []))
    if keypoints.ndim != 3 or len(keypoints) == 0:
        return 0, 0
    scores = np.asarray(cache.get('keypoint_conf', []))
    valid_xy = (keypoints[..., 0] > 0) & (keypoints[..., 1] > 0)
    if scores.shape == valid_xy.shape:
        valid_xy &= scores >= POSE_KEYPOINT_CONF
    return int(len(keypoints)), int(np.count_nonzero(valid_xy))


def draw_pose_skeleton(img, keypoints, keypoint_conf=None, transform=None):
    if keypoints is None or len(keypoints) == 0:
        return img
    points = np.asarray(keypoints)
    scores = None if keypoint_conf is None else np.asarray(keypoint_conf)
    for person_idx, person_points in enumerate(points):
        if len(person_points) < 17:
            continue
        person_scores = scores[person_idx] if scores is not None and person_idx < len(scores) else None
        visible = []
        for point_idx, point in enumerate(person_points):
            confidence = float(person_scores[point_idx]) if person_scores is not None and point_idx < len(person_scores) else 1.0
            if confidence < POSE_KEYPOINT_CONF or float(point[0]) <= 0 or float(point[1]) <= 0:
                visible.append(None)
                continue
            x, y = float(point[0]), float(point[1])
            if transform is not None:
                x, y = transform(x, y)
            visible.append((int(round(x)), int(round(y))))
        for start, end in COCO_POSE_SKELETON:
            if visible[start] is not None and visible[end] is not None:
                cv2.line(img, visible[start], visible[end], (0, 210, 255), 2, cv2.LINE_AA)
        for point in visible:
            if point is not None:
                cv2.circle(img, point, 3, (70, 70, 255), -1, cv2.LINE_AA)
    return img


def active_inference_model():
    return pose_model if pose_enabled and pose_model is not None else model


def active_model_ready():
    return pose_model_ready if pose_enabled else model_warmed


def current_recognition_name():
    return "\u4eba\u4f53\u59ff\u6001" if pose_enabled else "\u4eba\u4f53\u8bc6\u522b"


def load_pose_model():
    """Load the lightweight YOLO pose model, downloading it on first use."""
    global pose_model, pose_load_error
    if pose_model is not None:
        return pose_model
    try:
        POSE_MODEL_PATH.parent.mkdir(exist_ok=True)
        if POSE_MODEL_PATH.exists():
            source = str(POSE_MODEL_PATH)
        else:
            source = POSE_MODEL_NAME
        print(f"Loading pose model: {source}")
        pose_model = YOLO(source)
        # Ultralytics downloads named assets into the current working directory.
        # Move that first-use download into models/ so subsequent starts are offline.
        downloaded = Path(source).resolve() if source == POSE_MODEL_NAME else None
        if downloaded is not None and downloaded.exists() and not POSE_MODEL_PATH.exists():
            try:
                downloaded.replace(POSE_MODEL_PATH)
            except Exception:
                pass
        pose_load_error = None
        return pose_model
    except Exception as exc:
        pose_load_error = str(exc)
        print(f"Pose model load failed: {exc}")
        return None


def is_valid_person_box(x1, y1, x2, y2):
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    return box_h >= MIN_PERSON_BOX_HEIGHT and (box_w * box_h) >= MIN_PERSON_BOX_AREA


def reset_auto_snap_state(camera_key=None):
    """Reset suppression state so a new camera/person can be captured at once."""
    global last_snap_target, last_snap_time
    if camera_key is None:
        auto_snap_states.clear()
    else:
        auto_snap_states.pop(camera_key, None)
    last_snap_target = None
    last_snap_time = 0.0


def _seen_match_key(state, track_id, cx, cy, w, h,
                    tol_ratio=0.6, size_ratio=4.0):
    """Match a current person box to recent tracking history."""
    seen = state.get('seen', {})
    # 1) 绮剧‘ track_id 浼樺厛锛屼絾蹇呴』鍑犱綍涓€鑷达紙浣嶇疆杩?+ 灏哄鐩歌繎锛?
    if track_id >= 0 and track_id in seen:
        info = seen[track_id]
        if w > 0 and h > 0:
            dcx = abs(cx - info.get('cx', cx)) / w
            dcy = abs(cy - info.get('cy', cy)) / h
            d = (dcx * dcx + dcy * dcy) ** 0.5
            iw, ih = info.get('w', 0), info.get('h', 0)
            if d <= tol_ratio and iw > 0 and ih > 0:
                sr = (w * h) / float(iw * ih)
                sr = max(sr, 1.0 / sr)
                if sr <= size_ratio:
                    return track_id
    # 2) 鍑犱綍鍖归厤锛氭壘涓績鏈€杩戜笖灏哄鐩歌繎鑰?
    best_k = None
    best_d = 1e9
    for k, info in seen.items():
        if w <= 0 or h <= 0:
            continue
        dcx = abs(cx - info.get('cx', cx)) / w
        dcy = abs(cy - info.get('cy', cy)) / h
        d = (dcx * dcx + dcy * dcy) ** 0.5
        if d > tol_ratio:
            continue
        iw, ih = info.get('w', 0), info.get('h', 0)
        if iw <= 0 or ih <= 0:
            continue
        sr = (w * h) / float(iw * ih)
        sr = max(sr, 1.0 / sr)
        if sr > size_ratio:
            continue
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


def _compute_is_new(state, track_id, cx, cy, w, h, now_time, absence_reset):
    """Internal helper."""
    key = _seen_match_key(state, track_id, cx, cy, w, h)
    if key is None:
        return True
    rec = state['seen'].get(key)
    if rec is None:
        return True
    return (now_time - rec.get('t', 0.0)) > absence_reset


def _motion_of(rec, cx, cy, bw, bh, fw, fh):
    """Internal helper."""
    if not rec:
        return 0.0, 0.0
    pcx, pcy = rec.get('cx', cx), rec.get('cy', cy)
    pw, ph = rec.get('w', bw), rec.get('h', bh)
    move = (((cx - pcx) / max(fw, 1)) ** 2 +
            ((cy - pcy) / max(fh, 1)) ** 2) ** 0.5
    pa = pw * ph
    ca = bw * bh
    size_change = abs(ca - pa) / max(pa, 1) if pa > 0 else 0.0
    return move, size_change


def maybe_auto_snapshot(source_frame, boxes, camera_key, source_label=None):
    """Capture the best person, with movement and periodic fallback triggers."""
    global last_snap_target, last_snap_time, snap_flash, _global_last_snap
    if (not detection_enabled or not auto_snap or paused or sel or not focus or
            focus_id < 0 or source_frame is None):
        return False

    now_time = time.time()
    state = auto_snap_states.setdefault(camera_key, {
        'last_time': 0.0,
        'last_target': None,
        'missing_since': None,
        'absence_reset': False,
        'seen': {},
        'last_any_seen': -1e9,
        'last_any_snap': -1e9,
    })
    # 鏈抚寮€濮嬫椂鐨勨€滄渶杩戜竴娆＄敾闈㈡湁浜衡€濇椂闂达紝鐢ㄤ簬鍒ゅ畾鈥滅敾闈㈡槸鍚﹀垰浠庢棤浜哄彉鏈変汉鈥?
    prev_any_seen = state.get('last_any_seen', -1e9)
    frame_h, frame_w = source_frame.shape[:2]
    xyxy_list, conf_list, cls_list, track_id_list = get_boxes_data(boxes)

    # 瑙ｆ瀽鍊欓€夛細鏈抚寮€濮嬫椂鐨?seen 浠呰鍙栵紝缁熶竴鍦ㄥ啓鍥為樁娈垫洿鏂帮紝閬垮厤鍚屽抚浜掔浉姹℃煋銆?
    # 姣忎釜鍊欓€夊瓨 (cid, conf, x1,y1,x2,y2, track_id, mkey, rec)
    #   mkey 鈥斺€?鍐欏洖鍓嶅尮閰嶅埌鐨勫凡鐭ョ洰鏍?key锛圢one=灞忓箷涓婃湭鍖归厤鍒板凡鐭ョ洰鏍囷級
    #   rec  鈥斺€?鍐欏洖鍓嶈宸茬煡鐩爣鐨?seen 璁板綍锛堢敤浜庣畻甯ч棿杩愬姩/涓婃鎷嶇収鏃堕棿锛?
    candidates = []
    seen_updates = []  # (mkey, cx, cy, bw, bh, rec)
    for i in range(len(xyxy_list)):
        cid = int(cls_list[i])
        if cid != focus_id:
            continue
        x1, y1, x2, y2 = map(int, xyxy_list[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        if not is_valid_person_box(x1, y1, x2, y2):
            continue
        conf = float(conf_list[i])
        if conf < PERSON_CONF:
            continue
        track_id = int(track_id_list[i]) if i < len(track_id_list) else -1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = x2 - x1, y2 - y1
        # 鍐欏洖鍓嶅尮閰嶏紙姝ゆ椂 seen 灏氭湭琚湰甯у啓鍥烇紝鍖归厤缁撴灉鍙潬锛?
        mkey = _seen_match_key(state, track_id, cx, cy, bw, bh)
        rec = state['seen'].get(mkey) if mkey is not None else None
        candidates.append((cid, conf, x1, y1, x2, y2, track_id, mkey, rec))
        seen_updates.append((mkey, cx, cy, bw, bh, rec))

    if not candidates:
        if state['missing_since'] is None:
            state['missing_since'] = now_time
        elif (not state['absence_reset'] and
              now_time - state['missing_since'] >= snap_absence_reset):
            # photo was less than the normal stationary-person interval ago.
            state['last_target'] = None
            state['absence_reset'] = True
        return False

    state['missing_since'] = None
    state['absence_reset'] = False
    state['last_any_seen'] = now_time  # 鏈抚鏈夌洰鏍囷紝鏇存柊鈥滅敾闈㈡湁浜衡€濇椂闂?

    # 闄愬埗 seen 瀛楀吀瑙勬ā锛岄伩鍏嶉暱浼氳瘽鏃犻檺澧為暱
    if len(state['seen']) > 200:
        cutoff = now_time - 600
        state['seen'] = {k: v for k, v in state['seen'].items()
                         if v.get('t', 0.0) >= cutoff}

    # 鏈抚鎵€鏈夊湪鍦虹洰鏍囧啓鍥炩€滃湪鍦烘椂闂?浣嶇疆鈥濓紙淇濈暀鍏朵笂娆℃媿鐓ф椂闂?last_snap锛?
    for mkey, cx, cy, bw, bh, rec in seen_updates:
        if mkey is None:
            mkey = ("g", int(cx // 40), int(cy // 40), int(max(bw, 1) // 40))
        prev_snap = rec.get('last_snap', -1e9) if rec else -1e9
        state['seen'][mkey] = {'t': now_time, 'cx': cx, 'cy': cy,
                               'w': bw, 'h': bh, 'last_snap': prev_snap}

    # 鐢婚潰鍒氫粠鏃犱汉鍙樻湁浜?= 姝や汉鈥滅獊鐒跺嚭鐜扳€?
    really_new = (now_time - prev_any_seen) > snap_absence_reset
    last_any_snap = state.get('last_any_snap', -1e9)

    # 鍦ㄢ€滃簲璇ユ媿鈥濈殑鍊欓€夐噷锛岄€夌疆淇″害+闈㈢Н鏈€楂樿€咃紙閬垮厤鍗曞抚澶氱洰鏍囪繛鎷嶅埛灞忥級
    best = None
    best_score = -1.0
    best_reason = ''
    for c in candidates:
        (cid, conf, x1, y1, x2, y2, track_id, mkey, rec) = c
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = x2 - x1, y2 - y1
        if mkey is not None and rec is not None:
            # 宸茬煡鐩爣锛氱敤鍏惰嚜韬笂娆℃媿鐓ф椂闂翠笌甯ч棿杩愬姩
            since_snap = now_time - rec.get('last_snap', -1e9)
            move, size_change = _motion_of(rec, cx, cy, bw, bh, frame_w, frame_h)
            action_large = (move > snap_action_move or
                            size_change > snap_action_size)
        else:
            # 鍖归厤涓嶅埌锛氱敾闈㈣繎鏈熶竴鐩存湁浜?-> 瑙嗕负鍚屼汉绉诲姩/鏂版潵鑰咃紝鎸夆€滃姩浣滃ぇ鈥濆鐞嗭紱
            # 鐢ㄦ憚鍍忓ご绾ф渶灏忛棿闅旂害鏉燂紝閬垮厤鎸佺画璧板姩鑰呰姣忓抚鍒や负鏂颁汉鑰岄珮棰戞姄鎷?
            since_snap = now_time - last_any_snap
            action_large = True
        periodic_due = since_snap >= snap_periodic_interval
        should = False
        reason = ''
        if really_new and mkey is None:
            # 1) 绐佺劧鍑虹幇 -> 绔嬪埢鎷嶏紙浠呭彈鍏ㄥ眬鏂颁汉鍐峰嵈绾︽潫锛?
            if now_time - _global_last_snap >= snap_new_cooldown:
                should = True
                reason = 'new'
        else:
            # 2) 鍔ㄤ綔澶?鍑嗗绂诲紑 -> 鎷嶏紙鍙楁渶灏忛棿闅旂害鏉燂紝闃茶繛缁埛灞忥級
            if action_large and since_snap >= snap_min_gap:
                should = True
                reason = 'action'
            # 3) 绋冲畾鍦ㄥ満 -> 姣忛殧鍛ㄦ湡鎷嶄竴寮犵暀瀛?
            elif periodic_due and since_snap >= snap_min_gap:
                should = True
                reason = 'periodic'
        if should:
            area = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
            score = conf + area
            if score > best_score:
                best_score = score
                best = c
                best_reason = reason

    if best is None:
        return False

    (cid, conf, x1, y1, x2, y2, track_id, mkey, rec) = best
    # 鏇存柊鎷嶇収鏃堕棿锛氭憚鍍忓ご绾?+ 宸茬煡鐩爣鑷韩鐨?key
    state['last_any_snap'] = now_time
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = x2 - x1, y2 - y1
    eff_key = mkey if mkey is not None else \
        ("g", int(cx // 40), int(cy // 40), int(max(bw, 1) // 40))
    if eff_key in state['seen']:
        state['seen'][eff_key]['last_snap'] = now_time

    margin = int((x2 - x1) * 0.15)
    crop = source_frame[max(0, y1 - margin):min(frame_h, y2 + margin),
                        max(0, x1 - margin):min(frame_w, x2 + margin)].copy()
    if crop.size == 0:
        return False
    label = get_cn(all_names[cid])
    if source_label:
        label = f"{source_label}_{label}"
    snap_frame = source_frame if snap_mode == "full" else crop
    if not save_snapshot(snap_frame, label, conf, full_frame=(snap_mode == "full")):
        return False

    state['last_time'] = now_time
    curr_target = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    state['last_target'] = curr_target
    last_snap_time = now_time
    last_snap_target = curr_target
    _global_last_snap = now_time
    snap_flash = 12
    # 鎶婅繖娆℃姄鎷嶈褰曡繘浣嶇疆璁板繂锛堣瑙夊寲鈥滆皝鈥濓級
    idx = _pos_index((x1 + x2) / 2, (y1 + y2) / 2, frame_w, frame_h)
    _pos_set_snap(camera_key, idx, last_snap_path)
    reason_cn = {'new': 'new', 'action': 'action', 'periodic': 'periodic'}.get(best_reason, best_reason)
    print(f"  [auto snapshot] {source_label or chr(99)+chr(97)+chr(109)} saved ({reason_cn}, person #{track_id})")
    return True


def pick_best_detection(frame, boxes, valid_cids):
    """
    Pick the best detection for snapshot:
    Priority: highest confidence, then largest area.
    Returns (cropped_frame, cn_label, conf) or None.
    """
    # 统一获取检测数据
    xyxy_list, conf_list, cls_list, _ = get_boxes_data(boxes)

    best = None
    best_score = -1
    h, w = frame.shape[:2]

    for i in range(len(xyxy_list)):
        cid = int(cls_list[i])
        if cid not in valid_cids:
            continue
        conf = conf_list[i]
        x1, y1, x2, y2 = map(int, xyxy_list[i])
        if not is_valid_person_box(x1, y1, x2, y2):
            continue
        area = (x2 - x1) * (y2 - y1)

        # Score: confidence * 100 + normalized area
        norm_area = area / (w * h)
        score = conf * 100 + norm_area * 10

        if score > best_score:
            best_score = score
            best = (cid, conf, x1, y1, x2, y2)

    if best is None:
        return None

    cid, conf, x1, y1, x2, y2 = best
    cn_label = get_cn(all_names[cid])

    # Crop with margin, clamp to frame
    margin = int((x2 - x1) * 0.15)
    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(w, x2 + margin)
    cy2 = min(h, y2 + margin)
    cropped = frame[cy1:cy2, cx1:cx2].copy()

    return cropped, cn_label, conf

def find_cid(cn_name):
    en = cn_to_en.get(cn_name, "")
    for cid, name in all_names.items():
        if name == en:
            return cid
    for cid, name in all_names.items():
        if en and en in name:
            return cid
    return -1

# ====== Camera ======
def get_cap_backends():
    """Return candidate camera backends for the current OS."""
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    elif system == "Darwin":  # macOS
        return [cv2.CAP_AVFOUNDATION, 0]
    elif system == "Linux":
        return [cv2.CAP_V4L2, 0]
    return [0]


def get_cap_backend():
    return get_cap_backends()[0]


def _read_valid_camera_frame(cap, min_valid=3, max_attempts=10, delay=0.025):
    """Require several real frames before accepting a camera index.

    Some Windows backends briefly report ``isOpened()`` for unavailable/phantom
    indexes. A single successful read is not enough and caused bogus cameras to
    enter multi-camera mode. Requiring consecutive non-empty frames keeps the
    device list stable without rejecting a legitimately dark scene.
    """
    valid = 0
    last_frame = None
    expected_shape = None
    for _ in range(max_attempts):
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None
        if ret and frame is not None and frame.size > 0 and frame.ndim == 3:
            shape = frame.shape[:2]
            if expected_shape is None or shape == expected_shape:
                valid += 1
                expected_shape = shape
                last_frame = frame
                if valid >= min_valid:
                    return last_frame
            else:
                valid = 1
                expected_shape = shape
                last_frame = frame
        else:
            valid = 0
        if delay:
            time.sleep(delay)
    return None


def try_open_camera(idx, verify=False):
    for backend in get_cap_backends():
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            # Virtual cameras and USB capture cards can need close to a second
            # before their first stable frames. A short probe made available
            # cameras randomly disappear from the remote selector after restart.
            probe = None if not verify else _read_valid_camera_frame(
                cap, min_valid=2, max_attempts=30, delay=0.04)
            if not verify or probe is not None:
                return cap, backend
        cap.release()
    return cv2.VideoCapture(), None


def get_camera_devices(active_index=None):
    """Enumerate cameras that can continuously deliver valid frames."""
    devices = []
    if active_index is not None:
        devices.append({
            'index': int(active_index),
            'label': f'\u6444\u50cf\u5934{active_index} (\u5f53\u524d)',
        })
    scan_max = max(1, int(_os.environ.get("YOLO_CAMERA_SCAN_MAX", "6")))
    for i in range(scan_max):
        if active_index is not None and i == int(active_index):
            continue
        test_cap, backend = try_open_camera(i, verify=True)
        if test_cap.isOpened():
            backend_name = test_cap.getBackendName() if backend is not None else "default"
            devices.append({
                'index': i,
                'label': f'\u6444\u50cf\u5934{i} ({backend_name})',
            })
        test_cap.release()
    devices.sort(key=lambda item: item['index'])
    return devices

def get_cam_names(active_index=None):
    devices = get_camera_devices(active_index=active_index)
    return [device['label'] for device in devices] if devices else [f'\u6444\u50cf\u5934{i}' for i in range(3)]


def open_cam(idx):
    cap, backend = try_open_camera(idx)
    target_w, target_h = 1280, 720
    # Capture directly in the same 16:9 size used by inference/streaming.  Many
    # USB cameras default to uncompressed 1920x1080 YUY2, which is often limited
    # to about 15 FPS. MJPG 1280x720 substantially reduces USB bandwidth and
    # latency while preserving the full, uncropped camera field of view.
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or target_w
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_h
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap_fps = 30
    if cap.isOpened():
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        backend_name = cap.getBackendName() if backend is not None else "default"
        print(f"Camera {idx}: {cap_w}x{cap_h} @ ~{cap_fps}fps ({backend_name})")
    return cap, cap_w, cap_h, cap_fps


def select_cam(idx):
    global cur, cap, cam_w, cam_h, cam_fps
    old_cur = cur
    old_cap = cap
    new, nw, nh, nfps = open_cam(idx)
    if not new.isOpened():
        new.release()
        print(f"切换失败，保持当前摄像头: {old_cur}")
        return False

    cap = new
    cam_w, cam_h, cam_fps = nw, nh, nfps
    cur = idx
    if old_cap is not None and old_cap is not cap:
        old_cap.release()
    reset_auto_snap_state()
    return True


def switch_cam(d, cam_names=None):
    global cur, cap, cam_w, cam_h, cam_fps
    old_cur = cur
    old_cap = cap
    candidate = (cur + d) % 10

    new, nw, nh, nfps = open_cam(candidate)
    if not new.isOpened():
        new.release()
        candidate = old_cur
        new, nw, nh, nfps = open_cam(candidate)
        if not new.isOpened():
            new.release()
            cur = old_cur
            return

    cap = new
    cam_w, cam_h, cam_fps = nw, nh, nfps
    cur = candidate
    if old_cap is not None and old_cap is not cap:
        old_cap.release()
    reset_auto_snap_state()

# ====== Mouse Clickable Buttons ======
class Button:
    """Internal helper."""
    def __init__(self, x, y, w, h, label, callback, color=(60, 60, 60),
                 text_color=(255, 255, 255), font_size=14):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.callback = callback
        self.color = color          # 目标（静止）颜色
        self.text_color = text_color
        self.font_size = font_size
        self.hover = False
        self.hover_amt = 0.0        # 平滑悬停强度 0..1
        self._text_patch, self._text_alpha = render_text_patch(
            self.label, self.font_size, self.text_color)

    def contains(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, img, press_amt=0.0):
        # 悬停高亮：在静止色与加亮色之间按 hover_amt 平滑插值
        base = np.array(self.color, dtype=np.float32)
        hover_col = np.clip(base + 34, 0, 255)
        col = base * (1 - self.hover_amt) + hover_col * self.hover_amt
        if press_amt > 0:
            col = np.clip(col + 60 * press_amt, 0, 255)   # 点击瞬间整体提亮
        col = tuple(int(c) for c in col)

        # 按下时轻微内缩，营造“被压下去”的回弹反馈
        pad = int(2 * press_amt)
        x0, y0 = self.x + pad, self.y + pad
        x1, y1 = self.x + self.w - pad, self.y + self.h - pad

        shadow = tuple(max(0, c - 18) for c in col)
        cv2.rectangle(img, (x0, y0 + 2), (x1, y1 + 2), shadow, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
        # 边框：悬停/点击时提亮，过渡更柔和
        border = (28, 28, 28)
        if self.hover_amt > 0.05 or press_amt > 0:
            bcol = (np.array([90, 90, 90], dtype=np.float32) * self.hover_amt
                    + np.array([28, 28, 28], dtype=np.float32) * (1 - self.hover_amt))
            border = tuple(int(c) for c in bcol)
        cv2.rectangle(img, (x0, y0), (x1, y1), border, 1)
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2 - 1
        return blit_text_patch(
            img, self._text_patch, self._text_alpha, (center_x, center_y))


class ButtonManager:
    """Internal helper."""
    def __init__(self):
        self.buttons = []
        self.flash = {}     # label -> 点击时刻(time.time())，用于脉冲动画

    def add(self, btn):
        self.buttons.append(btn)

    def clear(self):
        self.buttons = []

    def handle_click(self, mx, my):
        for btn in self.buttons:
            if btn.contains(mx, my):
                self.flash[btn.label] = time.time()  # 触发点击脉冲
                btn.callback()
                return True
        return False

    def update_hover(self, mx, my):
        for btn in self.buttons:
            btn.hover = btn.contains(mx, my)

    def tick_animations(self, now):
        """每帧推进悬停平滑插值，并清理过期的点击脉冲。"""
        for btn in self.buttons:
            target = 1.0 if btn.hover else 0.0
            btn.hover_amt += (target - btn.hover_amt) * 0.28  # 鎸囨暟骞虫粦
            if btn.hover_amt < 0.01:
                btn.hover_amt = 0.0
        expired = [k for k, t0 in self.flash.items() if now - t0 > 0.4]
        for k in expired:
            del self.flash[k]

    def _flash_pulse(self, label, now):
        t0 = self.flash.get(label)
        if t0 is None:
            return 0.0
        dt = now - t0
        if dt > 0.4:
            return 0.0
        return max(0.0, 1.0 - dt / 0.4)  # 0.4s 内线性衰减

    def draw_all(self, img, now):
        for btn in self.buttons:
            pa = self._flash_pulse(btn.label, now)
            img = btn.draw(img, pa)
        return img


# ====== 摄像头切换过渡动画 ======
cam_transition = {'active': False, 't0': 0.0, 'dur': 0.4, 'old': None}

def start_cam_transition():
    """切换摄像头/多摄模式前调用：冻结当前画面作为淡出底图，
    新画面就绪后以交叉淡入方式平滑覆盖，消除硬切造成的卡顿感。"""
    global cam_transition
    old = g_frame.copy() if g_frame is not None else None
    cam_transition = {'active': True, 't0': time.time(), 'dur': 0.4, 'old': old}


# ====== Multi-Camera (澶氭憚骞抽摵) ======
def calc_grid(n):
    """根据摄像头数量计算最优行列布局 (rows, cols)。"""
    if n <= 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    elif n <= 4:
        return 2, 2
    elif n <= 6:
        return 2, 3
    elif n <= 9:
        return 3, 3
    else:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return rows, cols


def fit_complete_on_blur(frame, target_w, target_h):
    """Fit the entire frame without cropping and fill unused space with a live blur."""
    fh, fw = frame.shape[:2]
    # Use a live average-color matte behind the complete frame. It avoids black
    # bars without cropping and is much cheaper than blurring every camera tile.
    mean_bgr = cv2.mean(frame)[:3]
    matte = tuple(int(max(12, min(96, c * 0.42))) for c in mean_bgr)
    canvas = np.full((target_h, target_w, 3), matte, dtype=np.uint8)

    scale = min(target_w / max(fw, 1), target_h / max(fh, 1))
    new_w = max(1, min(target_w, int(round(fw * scale))))
    new_h = max(1, min(target_h, int(round(fh * scale))))
    fitted = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ox, oy = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = fitted
    return canvas, scale, ox, oy, new_w, new_h


class CamCaptureThread(threading.Thread):
    """Internal helper."""
    def __init__(self, cap, idx, label):
        super().__init__(name=f"camera-{idx}", daemon=True)
        self.cap = cap
        self.idx = idx
        self.label = label
        self._latest = None
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._times = collections.deque(maxlen=30)
        self.fps = 0.0
        self.error = False
        self._released = False

    def run(self):
        while not self._stop_event.is_set():
            try:
                t = time.time()
                # release() from another thread while DirectShow is inside read()
                # can terminate the whole Python process. Serialize both calls.
                with self._cap_lock:
                    if self._stop_event.is_set():
                        break
                    ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None
            if ret and frame is not None and frame.size > 0:
                with self._lock:
                    self._latest = frame
                self.error = False
                self._times.append(t)
                n = len(self._times)
                if n >= 2:
                    self.fps = (n - 1) / max(self._times[-1] - self._times[0], 1e-6)
            else:
                self.error = True
                self._stop_event.wait(0.02)

    def get_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def request_stop(self):
        self._stop_event.set()

    def release_after_join(self):
        if self.is_alive() or self._released:
            return False
        with self._cap_lock:
            try:
                self.cap.release()
            except Exception:
                pass
            self._released = True
        return True


    def take_capture_after_join(self):
        """Transfer ownership back to single-camera mode after the thread exits."""
        if self.is_alive() or self._released:
            return None
        self._released = True
        return self.cap


def open_all_cameras(reuse_capture=None, reuse_index=None):
    """启动多摄线程；复用当前单摄句柄，只打开其余摄像头。"""
    global multi_threads, multi_cached_results
    if multi_threads and not close_all_cameras():
        return False

    opened = []
    for dev in list(cam_devices):
        idx = dev['index']
        if reuse_capture is not None and idx == reuse_index:
            continue
        camera_cap, _ = try_open_camera(idx)
        if not camera_cap.isOpened():
            continue
        # Do not renegotiate FOURCC/height/FPS here: each DirectShow property can
        # rebuild the graph and add 1-2 seconds. Native HD is resized only for UI.
        try:
            camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if _read_valid_camera_frame(camera_cap, min_valid=2, max_attempts=6,
                                    delay=0.015) is None:
            camera_cap.release()
            print(f"Multi camera: camera {idx} has no valid frames; skipped")
            continue
        opened.append(CamCaptureThread(camera_cap, idx, dev['label']))

    total = len(opened) + (1 if reuse_capture is not None and reuse_capture.isOpened() else 0)
    if total < 2:
        for th in opened:
            th.request_stop()
            th.release_after_join()
        print(f"Multi camera: only {total} valid camera(s); need at least 2")
        return False

    if reuse_capture is not None and reuse_capture.isOpened():
        label = next((d['label'] for d in cam_devices if d['index'] == reuse_index),
                     f'\u6444\u50cf\u5934{reuse_index}')
        opened.insert(0, CamCaptureThread(reuse_capture, reuse_index, label))

    multi_threads = opened
    multi_cached_results = [empty_inference_cache() for _ in opened]
    for th in multi_threads:
        th.start()
    print(f"Multi camera opened: {len(multi_threads)} capture threads")
    return True


def close_all_cameras(timeout=4.0, keep_index=None):
    """安全停止全部线程；可把指定摄像头句柄直接交还单摄模式。"""
    global multi_threads, multi_cached_results
    threads = list(multi_threads)
    if not threads:
        multi_cached_results = []
        return None if keep_index is not None else True

    for th in threads:
        th.request_stop()
    deadline = time.monotonic() + timeout
    for th in threads:
        th.join(max(0.0, deadline - time.monotonic()))

    stuck = [th for th in threads if th.is_alive()]
    if stuck:
        print("多摄: 摄像头线程停止超时，已取消切换以保护程序: " +
              ", ".join(str(th.idx) for th in stuck))
        return False

    kept_capture = None
    for th in threads:
        if keep_index is not None and th.idx == keep_index and kept_capture is None:
            kept_capture = th.take_capture_after_join()
        else:
            th.release_after_join()
    multi_threads = []
    multi_cached_results = []
    return kept_capture if keep_index is not None else True


def read_all_frames():
    """取每路摄像头的最新帧，未就绪则给黑帧。"""
    frames = []
    for th in multi_threads:
        f = th.get_frame()
        if f is None:
            f = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(f)
    return frames


_no_signal_frame_cache = None


def make_no_signal_frame(w=None, h=None):
    """Create a cached no-signal placeholder frame."""
    global _no_signal_frame_cache
    w = OUTPUT_W if w is None else int(w)
    h = OUTPUT_H if h is None else int(h)
    if (_no_signal_frame_cache is not None and
            _no_signal_frame_cache.shape[:2] == (h, w)):
        return _no_signal_frame_cache.copy()
    img = np.full((h, w, 3), (18, 20, 26), dtype=np.uint8)
    cv2.putText(img, "CAMERA OFFLINE", (max(20, w // 2 - 190), h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 230, 240), 2, cv2.LINE_AA)
    cv2.putText(img, "RECONNECTING...", (max(20, w // 2 - 145), h // 2 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 190, 240), 2, cv2.LINE_AA)
    _no_signal_frame_cache = img
    return img.copy()


def do_toggle_multi_cam():
    """稳定切换多摄/单摄；过滤连点并串行化所有摄像头资源变更。"""
    global multi_cam, _buttons_built_for_size, cap, cur, cam_w, cam_h, cam_fps
    global multi_frame_counter, camera_switching, last_multi_toggle_at

    now = time.monotonic()
    if now - last_multi_toggle_at < MULTI_TOGGLE_DEBOUNCE:
        print("Multi camera toggle ignored: debounce")
        return False
    if camera_scan_running:
        print("Multi camera unavailable while camera scan is running")
        return False
    if not camera_transition_lock.acquire(blocking=False):
        print("Multi camera switch already in progress")
        return False

    camera_switching = True
    _buttons_built_for_size = None
    try:
        if not multi_cam:
            if len(cam_devices) < 2:
                print("Multi camera requires at least two cameras")
                return False
            start_cam_transition()
            if not open_all_cameras(reuse_capture=cap, reuse_index=cur):
                print("Multi camera open failed")
                return False
            multi_cam = True
            multi_frame_counter = 0
            reset_auto_snap_state()
            print(f"Multi camera mode: {len(multi_threads)} cameras")
        else:
            start_cam_transition()
            kept_cap = close_all_cameras(keep_index=cur)
            if kept_cap is False:
                print("Multi camera threads did not stop safely")
                return False
            if kept_cap is None or not kept_cap.isOpened():
                print("Camera handle transfer failed; reopening single camera")
                kept_cap, cam_w, cam_h, cam_fps = open_cam(cur)
                if not kept_cap.isOpened():
                    kept_cap.release()
                    print("Single camera recovery failed; reopening multi camera")
                    if open_all_cameras():
                        multi_cam = True
                    return False
            cap = kept_cap
            cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or OUTPUT_W
            cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or OUTPUT_H
            cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            multi_cam = False
            reset_auto_snap_state()
            print("Returned to single camera mode")
        return True
    except Exception as exc:
        print(f"Multi camera switch failed: {exc}")
        if not multi_cam and (cap is None or not cap.isOpened()):
            cap, cam_w, cam_h, cam_fps = open_cam(cur)
        return False
    finally:
        last_multi_toggle_at = time.monotonic()
        camera_switching = False
        _buttons_built_for_size = None
        camera_transition_lock.release()


# ====== 鎬ц兘浼樺寲閰嶇疆 ======
# 推理输入尺寸：GPU 算力足用较大尺寸提升小目标识别率；CPU 适当降低保速度
INFER_SIZE = 640
# 跳帧推理：GPU 逐帧推理(=1)以保证实时性与识别率；CPU 每 2 帧推理一次
INFER_EVERY_N = 1 if device == "cuda" else 2
POSE_INFER_EVERY_N = 2 if device == "cuda" else 3
# 多摄每路推理尺寸（每帧要跑多路，故偏小保证整体流畅）
MULTI_INFER_SIZE = 416 if device == "cuda" else 320
POSE_INFER_SIZE = 448 if device == "cuda" else 384
MULTI_POSE_INFER_SIZE = 352 if device == "cuda" else 288
# 使用半精度推理（需 GPU 支持）
USE_HALF = device == "cuda"
OUTPUT_W, OUTPUT_H = 1280, 720
# 妗岄潰绐楀彛灏哄绾︽潫锛氶攣瀹?16:9 鐢婚潰姣斾緥锛屽苟闄愬埗鏈€灏忓昂瀵革紝
# 閬垮厤鐢ㄦ埛鎷栨嫿绐楀彛鏃剁敾闈?鎸夐挳琚媺浼稿彉褰紝鎴栫缉寰楀お灏忕湅涓嶆竻銆?
DISPLAY_ASPECT = OUTPUT_W / OUTPUT_H   # 16:9
MIN_WIN_W = 640                        # 鏈€灏忕獥鍙ｅ锛堜笉灏忎簬姝ゅ€兼墠鐪嬪緱娓呯敾闈級
MIN_WIN_H = 360                        # 涓?MIN_WIN_W 鍚屾瘮渚?16:9
WIN_RESIZE_TOL = 2                     # 鏍℃瀹瑰樊(px)锛岄伩鍏嶆瘡甯ф姈鍔?姝诲惊鐜?
MULTI_TOP_BAR_H = 54
MULTI_CONTROL_W = 112
MULTI_TILE_INFO_H = 46
PHONE_STACK_W = 960      # 手机端多摄竖排时每路画面的宽度（竖排避免平铺过小）
PHONE_INFO_PANEL = _os.environ.get("YOLO_PHONE_PANEL", "0").lower() in {"1", "true", "yes", "on"}
STREAM_FPS = int(_os.environ.get("YOLO_STREAM_FPS", "20"))
STREAM_QUALITY = int(_os.environ.get("YOLO_STREAM_QUALITY", "76"))
STREAM_MAX_WIDTH = int(_os.environ.get("YOLO_STREAM_WIDTH", "1024"))



class LatestPoseWorker:
    """Internal helper."""
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame = None
        self._result = empty_inference_cache()
        self._running = True
        self.inference_ms = 0.0
        self.completed = 0
        self._thread = threading.Thread(target=self._loop, name="pose-inference", daemon=True)
        self._thread.start()

    def submit(self, frame):
        if frame is None:
            return
        with self._lock:
            self._frame = frame.copy()
        self._event.set()

    def latest(self):
        with self._lock:
            return self._result

    def reset(self):
        with self._lock:
            self._frame = None
            self._result = empty_inference_cache()
        self.inference_ms = 0.0
        self.completed = 0

    def _loop(self):
        while self._running:
            self._event.wait(0.5)
            self._event.clear()
            if not self._running:
                break
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None or not pose_enabled or not pose_model_ready or pose_model is None:
                continue
            t0 = time.perf_counter()
            try:
                results = pose_model.predict(
                    frame, imgsz=POSE_INFER_SIZE, verbose=False, half=USE_HALF,
                    classes=[0], conf=PERSON_CONF, iou=PERSON_IOU,
                )
                cache = result_to_cache(results[0])
                elapsed = (time.perf_counter() - t0) * 1000.0
                with self._lock:
                    self._result = cache
                self.inference_ms = elapsed
                self.completed += 1
            except Exception as exc:
                if self.completed % 60 == 0:
                    print(f"Async pose inference failed: {exc}")

    def stop(self):
        self._running = False
        self._event.set()


pose_async = LatestPoseWorker()


_phone_snap_cache = (None, None)  # (path, thumb) 抓拍缩略图缓存


def get_snap_thumb():
    """返回最近抓拍的缩略图(140x94 BGR)，带路径缓存；无则返回 None。"""
    global _phone_snap_cache
    if last_snap_path is None:
        return None
    if _phone_snap_cache[0] == last_snap_path:
        return _phone_snap_cache[1]
    try:
        img = cv2.imread(last_snap_path)
        if img is None:
            return None
        th, tw = img.shape[:2]
        scale = min(140 / tw, 94 / th)
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        timg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        pad = np.full((94, 140, 3), (20, 22, 28), np.uint8)
        ox, oy = (140 - nw) // 2, (94 - nh) // 2
        pad[oy:oy + nh, ox:ox + nw] = timg
        _phone_snap_cache = (last_snap_path, pad)
        return pad
    except Exception:
        return None


def make_phone_portrait(cam, person_count=0, fps=0.0, locked_id=None,
                        locked_conf=0.0, snap_thumb=None, online=0):
    """Build a complete, uncropped portrait monitor frame."""
    W, H = 720, 1280
    ch, cw = cam.shape[:2]
    cam_h = max(1, int(round(W * ch / max(cw, 1))))
    cam_r = cv2.resize(cam, (W, cam_h), interpolation=cv2.INTER_LINEAR)
    top = max(0, (H - cam_h) // 2)
    tiny = cv2.resize(cam, (60, 106), interpolation=cv2.INTER_AREA)
    tiny = cv2.GaussianBlur(tiny, (0, 0), 3.5)
    canvas = cv2.resize(tiny, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.addWeighted(canvas, 0.34, np.zeros_like(canvas), 0.66, 0)
    end = min(H, top + cam_h)
    canvas[top:end, :W] = cam_r[:end-top]
    now = datetime.datetime.now()
    cv2.putText(canvas, "NEURAL WATCH", (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, (80, 245, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, now.strftime("%Y-%m-%d %H:%M:%S"), (350, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 240), 1, cv2.LINE_AA)
    info_y = min(H - 90, end + 40)
    cv2.putText(canvas, f"PEOPLE {person_count}   {fps:.1f} FPS   ONLINE {online}",
                (24, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (110, 220, 255), 2, cv2.LINE_AA)
    if locked_id is not None:
        cv2.putText(canvas, f"LOCK #{locked_id}  {locked_conf:.2f}", (24, info_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 245, 170), 2, cv2.LINE_AA)
    if snap_thumb is not None and info_y + 130 < H:
        canvas[info_y + 18:info_y + 112, W - 164:W - 24] = snap_thumb
    return canvas

# ====== State ======
cur = 0
cap, cam_w, cam_h, cam_fps = open_cam(cur)
if not cap.isOpened():
    print("Cannot open camera")
    input()
    exit(1)
# Do not scan camera indexes 0..9 during startup: unavailable DirectShow/MSMF
# devices can each block for hundreds of milliseconds and make the app appear dead.
# The Refresh button still performs a full scan on demand.
cam_devices = [{'index': cur, 'label': f'\u6444\u50cf\u5934{cur}'}]
cam_names = [cam_devices[0]['label']]
camera_scan_lock = threading.Lock()
camera_scan_running = False
camera_transition_lock = threading.Lock()
camera_switching = False
last_multi_toggle_at = 0.0
MULTI_TOGGLE_DEBOUNCE = 0.8
pending_actions_lock = threading.Lock()

win = "YOLOv8"
full = False
focus = True           # 默认专注模式：只识别人
focus_cn = PERSON_CN
focus_id = find_cid(PERSON_CN)
detection_enabled = not (_os.environ.get("YOLO_START_MONITOR", "0").lower() in {"1", "true", "yes", "on"})  # False = pure monitoring
pose_enabled = False       # True = use YOLO pose keypoints instead of box-only detection
inp = ""
sel = False
sel_page = 0
paused = False
show_help = False
auto_snap = False  # 默认关闭自动拍照，避免开机即抓拍
# ===== 鑷姩鎶撴媿鑺傚鎺у埗锛堣瑙?maybe_auto_snapshot锛?====
# 绐佺劧鍑虹幇鐨勬柊浜轰箣闂寸殑鏈€鐭叏灞€闂撮殧(绉?锛屼粎闃插悓甯у浜哄埛灞忥紝涓嶅奖鍝嶁€滅珛鍒绘媿鈥濊涔?
snap_new_cooldown = float(_os.environ.get("YOLO_SNAP_NEW_COOLDOWN", "1.0"))
# 鍚屼竴鐩爣涓ゆ鎷嶇収鐨勬渶灏忛棿闅?绉?锛氶槻姝⑩€滃姩浣滃ぇ鈥濇椂杩炵画澶氬抚鍒峰睆
snap_min_gap = float(_os.environ.get("YOLO_SNAP_MIN_GAP", "8.0"))
# 绋冲畾鍦ㄥ満(浣嶇疆闄勮繎銆佸熀鏈笉鍔?鏃跺懆鏈熺暀瀛樻媿鐨勯棿闅?绉?锛氶殧涓€娈垫媿涓€寮?
snap_periodic_interval = float(_os.environ.get("YOLO_SNAP_PERIODIC", "45.0"))
# 甯ч棿涓績浣嶇Щ鍗犵敾闈㈡瘮渚嬶紝瓒呰繃瑙嗕负鈥滃姩浣滃箙搴﹀彉澶р€?
snap_action_move = float(_os.environ.get("YOLO_SNAP_ACTION_MOVE", "0.18"))
# 灏哄(闈㈢Н)鍙樺寲姣斾緥锛岃秴杩囪涓衡€滃姩浣滃箙搴﹀彉澶?鍑嗗绂诲紑鈥?
snap_action_size = float(_os.environ.get("YOLO_SNAP_ACTION_SIZE", "0.40"))
# 浜虹寮€瓒呰繃璇ョ鏁颁笖鍐嶆鍥炴潵锛岃涓烘柊浜嬩欢锛堥噸鏂板彲绔嬪嵆鎷嶏級
snap_absence_reset = float(_os.environ.get("YOLO_SNAP_ABSENCE_RESET", "3"))
last_snap_time = 0.0
snap_flash = 0
snap_mode = "crop"   # "crop" = 拍检测物体, "full" = 全屏拍照(带水印)
last_snap_path = None  # 最近一次抓拍的绝对路径（用于手机端底部缩略图）

# ====== 手机同步观看（局域网 MJPEG 推流）======
remote_url = None
remote_short_url = None
qr_target = "public"
STREAM_PORT = 8000
streamer = MJPEGStreamer(port=STREAM_PORT, quality=STREAM_QUALITY, fps_cap=STREAM_FPS,
                         max_width=STREAM_MAX_WIDTH, max_height=720,
                         on_command=phone_command,
                         file_root=REC_DIR, snap_root=SNAP_DIR,
                         get_file_list=list_playback_files,
                         get_status=lambda: {"recording": recording,
                                             "clients": streamer.client_count(),
                                             "remote": remote_url,
                                             "short_url": remote_short_url,
                                             "display_url": remote_short_url or remote_url or streamer.url,
                                             "lan": streamer.url,
                                             "detection_enabled": detection_enabled,
                                             "pose_enabled": pose_enabled,
                                             "pose_ready": pose_model_ready,
                                             "pose_loading": pose_model_loading,
                                             "pose_error": pose_load_error,
                                             "pose_people": pose_people_count,
                                             "pose_keypoints": pose_visible_keypoints,
                                             "pose_inference_ms": pose_async.inference_ms,
                                             "detected_people": detected_people_count,
                                             "recognition_mode": "pose" if pose_enabled else ("detect" if detection_enabled else "monitor"),
                                             "auto_snap": auto_snap,
                                             "snapshot_count": snapshot_count,
                                             "last_snapshot": Path(last_snap_path).name if last_snap_path else None,
                                             "snapshot_dir": str(SNAP_DIR),
                                            "camera_switching": camera_switching,
                                            "camera_count": len(cam_devices),
                                            "camera_indices": [d['index'] for d in cam_devices],
                                            "camera_labels": [d['label'] for d in cam_devices],
                                            "current_camera": cur,
                                            "multi_camera": multi_cam,
                                             "active_camera_count": len(multi_threads) if multi_cam else 1,
                                             "qr_target": qr_target,
                                             "qr_url": ((remote_short_url or remote_url) if qr_target == "public" and remote_url else streamer.url),
                                             "stream_fps": streamer.encoded_fps,
                                             "encode_ms": streamer.encode_ms,
                                             "jpeg_kb": streamer.jpeg_kb,
                                             "stream_width": streamer.max_width,
                                             "stream_quality": streamer.quality,
                                             "stream_fps_cap": streamer.fps_cap,
                                             "stream_profile": streamer.profile_name,
                                             "pos_grid": {"cols": POS_GRID_COLS, "rows": POS_GRID_ROWS},
                                             "position_map": (get_position_map_summary(("camera", cur))
                                                              if not multi_cam else None),
                                             "position_maps": ({th.label: get_position_map_summary(("camera", th.idx))
                                                                for th in multi_threads}
                                                              if multi_cam else None)})
streamer.start()
show_stream_qr = True   # 是否在窗口上叠加二维码/地址
_qr_img = make_qr_image(streamer.url)  # 连上前先用局域网二维码；连上后自动切公网
stream_url = streamer.url


def _try_shorten_public_url(long_url):
    """Best-effort public URL shortening; never block tunnel availability."""
    from urllib.parse import quote, urlencode
    from urllib.request import Request, urlopen

    # cleanuri currently works reliably on this network and returns JSON.
    try:
        payload = urlencode({"url": long_url}).encode("ascii")
        req = Request(
            "https://cleanuri.com/api/v1/shorten",
            data=payload,
            headers={
                "User-Agent": "YOLO-Remote-Monitor/1.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read(2048).decode("utf-8", errors="ignore"))
        candidate = str(data.get("result_url", "")).replace("\/", "/").strip()
        if candidate.startswith(("https://", "http://")) and len(candidate) < len(long_url):
            return candidate
    except Exception:
        pass

    providers = (
        "https://is.gd/create.php?format=simple&url=" + quote(long_url, safe=""),
        "https://tinyurl.com/api-create.php?url=" + quote(long_url, safe=""),
    )
    for api in providers:
        try:
            req = Request(api, headers={"User-Agent": "YOLO-Remote-Monitor/1.0"})
            with urlopen(req, timeout=3.5) as resp:
                candidate = resp.read(512).decode("utf-8", errors="ignore").strip()
            if candidate.startswith(("https://", "http://")) and len(candidate) < len(long_url):
                return candidate
        except Exception:
            continue
    return None

def set_remote_url(u):
    """隧道建立后回调：记录公网地址，并在二维码指向公网时刷新。"""
    global remote_url, remote_short_url, _qr_img, qr_target
    remote_url = u
    remote_short_url = None
    print("\n[Public] Remote URL: " + u)
    if qr_target == "public":
        _qr_img = make_qr_image(u)

    def short_worker(expected_url=u):
        global remote_short_url, _qr_img
        short = None
        for attempt in range(3):
            short = _try_shorten_public_url(expected_url)
            if short or remote_url != expected_url:
                break
            time.sleep(3.0 + attempt * 2.0)
        if short and remote_url == expected_url:
            remote_short_url = short
            print("[Public] Short URL: " + short)
            if qr_target == "public":
                _qr_img = make_qr_image(short)

    threading.Thread(target=short_worker, name="public-url-shortener", daemon=True).start()


def do_toggle_stream_qr():
    global show_stream_qr, _buttons_built_for_size
    _buttons_built_for_size = None
    show_stream_qr = not show_stream_qr


def retry_tunnel():
    """（重新）启动公网隧道。按 B 且尚未连上时调用，真正重试而非仅提示。"""
    global tunnel, remote_url
    if tunnel is not None:
        try:
            tunnel.stop()
        except Exception:
            pass
        tunnel = None
    try:
        import remote_access
        t = remote_access.start_tunnel(STREAM_PORT, on_url=set_remote_url)
        if t is None:
            print("[公网] 仍未找到 ssh，无法建立隧道（详见控制台提示）")
        else:
            tunnel = t
            print("[公网] 隧道已重新启动，等待中继返回地址（见控制台）...")
    except Exception as e:
        print(f"[公网] 重建隧道失败: {e}")


def do_toggle_qr_target():
    """在『局域网二维码 / 公网二维码』之间切换；尚未连上时按 B 会先尝试建立隧道。"""
    global qr_target, _qr_img, _buttons_built_for_size
    _buttons_built_for_size = None
    if remote_url is None:
        retry_tunnel()
        return
    qr_target = "public" if qr_target == "lan" else "lan"
    _qr_img = make_qr_image((remote_short_url or remote_url) if qr_target == "public" else streamer.url)
    print(f"[二维码] 指向: {qr_target}")


# ====== Remote tunnel ======
if _os.environ.get("YOLO_DISABLE_TUNNEL", "0").lower() in {"1", "true", "yes", "on"}:
    tunnel = None
    print("Remote tunnel disabled by YOLO_DISABLE_TUNNEL")
else:
    try:
        import remote_access
        tunnel = remote_access.start_tunnel(STREAM_PORT, on_url=set_remote_url)
        if tunnel is None:
            print("[Remote] Tunnel unavailable; LAN viewing remains available")
    except Exception as e:
        tunnel = None
        print(f"[Remote] Tunnel startup failed: {e}")

# Warm the detector in a worker thread so the camera preview appears immediately.
model_warmed = False
model_warming = False
model_warm_error = None
model_warm_lock = threading.Lock()


def start_model_warm_async():
    global model_warming
    with model_warm_lock:
        if model_warmed or model_warming:
            return
        model_warming = True

    def worker():
        global model_warmed, model_warming, model_warm_error
        try:
            print("Warming up model in background...")
            dummy = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
            model.track(
                dummy,
                imgsz=INFER_SIZE,
                verbose=False,
                half=USE_HALF,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[focus_id],
                conf=PERSON_CONF,
                iou=PERSON_IOU,
            )
            with model_warm_lock:
                model_warmed = True
                model_warm_error = None
            print("Model ready.")
        except Exception as exc:
            with model_warm_lock:
                model_warm_error = str(exc)
            print(f"Model warmup failed: {exc}")
        finally:
            with model_warm_lock:
                model_warming = False

    threading.Thread(target=worker, name="model-warmup", daemon=True).start()


def start_pose_warm_async():
    """Load/download and warm the pose model without blocking camera preview."""
    global pose_model_loading, pose_load_error
    with pose_load_lock:
        if pose_model_ready or pose_model_loading:
            return
        pose_model_loading = True
        pose_load_error = None

    def worker():
        global pose_model_ready, pose_model_loading, pose_load_error
        try:
            pm = load_pose_model()
            if pm is None:
                raise RuntimeError(pose_load_error or "Pose model load failed")
            print("Warming up pose model in background...")
            dummy = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
            pm.predict(
                dummy,
                imgsz=POSE_INFER_SIZE,
                verbose=False,
                half=USE_HALF,
                classes=[0],
                conf=PERSON_CONF,
                iou=PERSON_IOU,
            )
            with pose_load_lock:
                pose_model_ready = True
                pose_load_error = None
            print("Pose model ready.")
        except Exception as exc:
            with pose_load_lock:
                pose_model_ready = False
                pose_load_error = str(exc)
            print(f"Pose model warmup failed: {exc}")
        finally:
            with pose_load_lock:
                pose_model_loading = False

    threading.Thread(target=worker, name="pose-model-warmup", daemon=True).start()


# Model warmup is deferred until the first inference, after the window is visible.

# 智能拍照：记录上次拍照的目标状态（位置+大小），用于判断是否需要补拍
last_snap_target = None  # (cx, cy, w, h) 最近一次自动抓拍目标
# 娉細鍔ㄤ綔/鍛ㄦ湡鎷嶇収闃堝€煎凡杩佺Щ鍒颁笂闈?snap_action_* / snap_periodic_interval 绛夊父閲?
auto_snap_states = {}       # 每个摄像头独立维护冷却、离场与周期补拍状态

# ====== 浣嶇疆鍗犵敤璁板繂锛堝骇浣嶈蹇嗭級======
# 鎶婄敾闈㈡寜缃戞牸鍒囧垎涓鸿嫢骞测€滀綅缃€濓紝璁板綍姣忎釜浣嶇疆褰撳墠鍧愮潃璋侊紙track_id锛夈€?
# 宸插仠鐣欏涔咃紱绌虹疆鏃惰浣忎笂涓€浣嶅崰鐢ㄨ€呫€備粎浠ヨ窡韪?id 鏍囪瘑鈥滆皝鈥濓紙鏃犱汉鑴歌瘑鍒級锛?
# 鏁呪€滆皝鈥濇寚绯荤粺鏈浼氳瘽涓殑浜虹墿缂栧彿锛涢噸鍚悗 id 浼氶噸缃紝浣嗏€滄煇浣嶇疆涓婃鍧愮殑鏄汉鐗?X鈥?
# 杩欎竴璁板繂浼氳鎸佷箙鍖栦繚鐣欍€?
POS_GRID_COLS = int(_os.environ.get("YOLO_POS_COLS", "4"))
POS_GRID_ROWS = int(_os.environ.get("YOLO_POS_ROWS", "2"))
POS_MEM_FILE = BASE_DIR / "position_memory.json"
POS_MEM_SAVE_INTERVAL = 10.0   # 鎸佷箙鍖栬妭娴侊紙绉掞級
position_memory = {}           # camera_key -> [ {track_id, since, last_track_id, last_seen, snap}, ... ]
_pos_mem_last_save = 0.0
_global_last_snap = 0.0        # 浠绘剰鎽勫儚澶存渶杩戜竴娆℃姄鎷嶆椂鍒伙紙鐢ㄤ簬鏂颁汉绐佸彂鍐峰嵈锛?


def _pos_index(cx, cy, fw, fh, cols=POS_GRID_COLS, rows=POS_GRID_ROWS):
    """Internal helper."""
    col = min(cols - 1, max(0, int(cx / max(fw, 1) * cols)))
    row = min(rows - 1, max(0, int(cy / max(fh, 1) * rows)))
    return row * cols + col


def _pos_key(ck):
    """Internal helper."""
    if isinstance(ck, (tuple, list)):
        return json.dumps(list(ck), ensure_ascii=False)
    return str(ck)


def _pos_load():
    """Internal helper."""
    global position_memory
    try:
        if POS_MEM_FILE.exists():
            with open(POS_MEM_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for raw_key, arr in data.items():
                try:
                    key = tuple(json.loads(raw_key))
                except Exception:
                    key = raw_key
                states = []
                for st in arr:
                    states.append({
                        'track_id': None,                       # 閲嶅惎鍚庡綋鍓嶅崰鐢ㄦ竻闆?
                        'since': 0.0,
                        'last_track_id': st.get('last_track_id'),
                        'last_seen': float(st.get('last_seen', 0.0) or 0.0),
                        'snap': st.get('snap'),
                    })
                position_memory[key] = states
    except Exception:
        pass


def _pos_mem_maybe_save(now):
    global _pos_mem_last_save
    if now - _pos_mem_last_save < POS_MEM_SAVE_INTERVAL:
        return
    _pos_mem_last_save = now
    try:
        serial = {}
        for ck, arr in position_memory.items():
            serial[_pos_key(ck)] = [
                {'track_id': s['track_id'],
                 'since': s['since'],
                 'last_track_id': s['last_track_id'],
                 'last_seen': s['last_seen'],
                 'snap': s['snap']}
                for s in arr
            ]
        with open(POS_MEM_FILE, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False)
    except Exception:
        pass


def update_position_memory(frame, boxes, camera_key):
    """Internal helper."""
    global position_memory
    if frame is None or boxes is None:
        return
    fh, fw = frame.shape[:2]
    xyxy_list, conf_list, cls_list, track_id_list = get_boxes_data(boxes)
    n = POS_GRID_COLS * POS_GRID_ROWS
    states = position_memory.setdefault(camera_key, [None] * n)
    if len(states) != n:
        states = [None] * n
        position_memory[camera_key] = states

    # 鏈抚鍚勭綉鏍肩殑鍗犵敤鑰咃紙鍚屾牸澶氫汉鍙栫疆淇″害鏈€楂橈級
    cell_occ = [None] * n
    for i in range(len(xyxy_list)):
        cid = int(cls_list[i])
        if focus and cid != focus_id:
            continue
        x1, y1, x2, y2 = map(int, xyxy_list[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if not is_valid_person_box(x1, y1, x2, y2):
            continue
        conf = float(conf_list[i])
        if conf < PERSON_CONF:
            continue
        tid = int(track_id_list[i]) if i < len(track_id_list) else -1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        idx = _pos_index(cx, cy, fw, fh)
        if cell_occ[idx] is None or conf > cell_occ[idx][1]:
            cell_occ[idx] = (tid, conf)

    now = time.time()
    for idx in range(n):
        st = states[idx]
        if st is None:
            st = {'track_id': None, 'since': 0.0,
                  'last_track_id': None, 'last_seen': 0.0, 'snap': None}
            states[idx] = st
        occ = cell_occ[idx]
        if occ is None:
            if st['track_id'] is not None:
                st['last_track_id'] = st['track_id']
                st['last_seen'] = now
                st['track_id'] = None
                st['since'] = 0.0
        else:
            tid, _ = occ
            if st['track_id'] != tid:
                if st['track_id'] is not None:
                    st['last_track_id'] = st['track_id']
                    st['last_seen'] = now
                st['track_id'] = tid
                st['since'] = now
    _pos_mem_maybe_save(now)


def _pos_set_snap(camera_key, idx, path):
    """Internal helper."""
    states = position_memory.get(camera_key)
    if states is None or idx < 0 or idx >= len(states):
        return
    if states[idx] is not None:
        states[idx]['snap'] = path


def get_position_map_summary(camera_key):
    """Internal helper."""
    states = position_memory.get(camera_key)
    if not states:
        return []
    now = time.time()
    out = []
    for idx, st in enumerate(states):
        if st is None:
            out.append({"pos": idx, "occupied": False, "track_id": None,
                        "dwell_s": 0, "last_track_id": None})
            continue
        occ = st['track_id'] is not None
        out.append({
            "pos": idx,
            "occupied": occ,
            "track_id": st['track_id'] if occ else None,
            "dwell_s": int(now - st['since']) if occ else 0,
            "last_track_id": st['last_track_id'],
        })
    return out


_pos_load()  # 鍚姩鏃舵仮澶嶄笂娆＄殑搴т綅璁板繂

# Latest detection results (updated each loop for snapshot functions)
g_frame = None
g_boxes = None

# 推理缓存：存储上一次推理结果，用于跳帧时复用
cached_boxes = None
frame_counter = 0
_first_inference_logged = False

# 锁定追踪：持续跟随同一个人，避免每帧换框
locked_track_id = None
locked_box = None
locked_conf = 0.0
locked_misses = 0
LOCK_MISS_TOLERANCE = 12

# ====== 澶氭憚鍍忓ご骞抽摵妯″紡 ======
multi_cam = False          # 是否处于多摄平铺模式
multi_threads = []         # [CamCaptureThread, ...] 每路独立采集线程
multi_cached_results = []  # 姣忚矾鎽勫儚澶寸殑鎺ㄧ悊缂撳瓨 [{boxes, conf, cls}, ...]
multi_frame_counter = 0    # 全局帧计数器（仅用于多摄跳帧）
MULTI_TILE_INFER_N = 2     # 多摄模式下每 N 帧推理一次（每路）

# 按钮缓存：避免每帧重建
_buttons_built_for_size = None
_buttons_cache = None

# Quick items
quick_items = [PERSON_CN]

btn_mgr = ButtonManager()


def toggle_full():
    global full, _buttons_built_for_size
    _buttons_built_for_size = None
    full = not full
    if not show_desktop or not window_created:
        return   # 后台隐藏运行时只记录状态，显示画面时再按 full 建窗
    _destroy_desktop_window()
    _create_desktop_window(full)


def do_switch_cam(d):
    global _buttons_built_for_size
    if multi_cam:
        return
    start_cam_transition()
    _buttons_built_for_size = None
    switch_cam(d, cam_names)


def do_select_cam(idx):
    global _buttons_built_for_size
    if multi_cam:
        return
    start_cam_transition()
    if select_cam(idx):
        _buttons_built_for_size = None


def do_refresh_cameras():
    """Refresh camera cards in a worker thread so the preview stays responsive."""
    global camera_scan_running, _buttons_built_for_size
    if multi_cam or camera_switching:
        return
    with camera_scan_lock:
        if camera_scan_running:
            return
        camera_scan_running = True
    _buttons_built_for_size = None

    def worker():
        global cam_devices, cam_names, camera_scan_running, _buttons_built_for_size
        try:
            found = get_camera_devices(active_index=cur)
            if found and not multi_cam and not camera_switching:
                cam_devices = found
                cam_names = [device['label'] for device in found]
            print(f"可用摄像头: {len(cam_devices)}")
        except Exception as exc:
            print(f"摄像头扫描失败: {exc}")
        finally:
            with camera_scan_lock:
                camera_scan_running = False
            _buttons_built_for_size = None
            _update_tray_menu()   # 鎵弿瀹屾垚鍚庡埛鏂版墭鐩樿彍鍗曠殑鎽勫儚澶村垪琛?

    threading.Thread(target=worker, name="camera-scan", daemon=True).start()

def clear_inference_state():
    global cached_boxes, g_boxes, locked_track_id, locked_box, locked_conf, locked_misses
    global multi_cached_results, _first_inference_logged, frame_counter, multi_frame_counter
    global pose_people_count, pose_visible_keypoints, detected_people_count
    cached_boxes = empty_inference_cache()
    g_boxes = cached_boxes
    multi_cached_results = [empty_inference_cache() for _ in multi_cached_results]
    locked_track_id = None
    locked_box = None
    locked_conf = 0.0
    locked_misses = 0
    frame_counter = 0
    multi_frame_counter = 0
    _first_inference_logged = False
    pose_people_count = 0
    pose_visible_keypoints = 0
    detected_people_count = 0
    pose_async.reset()
    reset_auto_snap_state()


def do_toggle_detection():
    global detection_enabled, auto_snap, _buttons_built_for_size
    detection_enabled = not detection_enabled
    clear_inference_state()
    if not detection_enabled:
        auto_snap = False
    elif pose_enabled:
        start_pose_warm_async()
    else:
        start_model_warm_async()
    _buttons_built_for_size = None
    print("Detection mode: ON" if detection_enabled else "Monitor mode: ON (inference disabled)")


def do_toggle_pose():
    global pose_enabled, detection_enabled, _buttons_built_for_size
    # Pose inference replaces box-only inference, so enabling it does not double
    # GPU workload. The pose model still provides person boxes for snapshots.
    if not pose_enabled and not detection_enabled:
        detection_enabled = True
    pose_enabled = not pose_enabled
    clear_inference_state()
    if detection_enabled:
        if pose_enabled:
            start_pose_warm_async()
        else:
            start_model_warm_async()
    _buttons_built_for_size = None
    state = "ON" if pose_enabled else "OFF"
    print(f"Pose recognition: {state}")


def do_toggle_focus():
    global _buttons_built_for_size
    _buttons_built_for_size = None
    print("Mode: PERSON ONLY")


def do_toggle_selector():
    global sel, inp, sel_page, _buttons_built_for_size
    _buttons_built_for_size = None
    sel = False
    inp = ""
    sel_page = 0
    print("Selector disabled: person-only mode")


def do_toggle_pause():
    global paused, _buttons_built_for_size
    _buttons_built_for_size = None
    paused = not paused
    print(f"Paused: {paused}")


def do_toggle_help():
    global show_help, _buttons_built_for_size
    _buttons_built_for_size = None
    show_help = not show_help


def do_toggle_auto_snap():
    global auto_snap, _buttons_built_for_size
    # 自动拍照依赖识别；从监控模式开启时自动恢复识别，避免按钮显示开启却永远不拍。
    if not auto_snap and not detection_enabled:
        do_toggle_detection()
    auto_snap = not auto_snap
    reset_auto_snap_state()
    _buttons_built_for_size = None
    print(f"Auto-snap: {'ON' if auto_snap else 'OFF'}")


def do_toggle_snap_mode():
    global snap_mode, _buttons_built_for_size
    _buttons_built_for_size = None
    snap_mode = "full" if snap_mode == "crop" else "crop"
    print(f"Snap mode: {'全屏拍照' if snap_mode == 'full' else '人物抓拍'}")


def do_manual_snap():
    """Manual snapshot: works regardless of target detection.
    Mode: crop = best detection object, full = entire frame with watermark."""
    global snap_flash
    if g_frame is None:
        print("  [拍照] 尚未获取到摄像头画面")
        return

    if snap_mode == "full":
        # Full frame with watermark, no detection needed
        save_snapshot(g_frame, "全屏", 1.0, full_frame=True)
        snap_flash = 12
    else:
        valid = {focus_id}
        xyxy_list, conf_list, cls_list, _ = get_boxes_data(g_boxes)
        if len(xyxy_list) == 0:
            # No detections: save full frame without watermark
            save_snapshot(g_frame, "no_target", 0.0)
            snap_flash = 12
            return
        # 创建临时的 boxes 对象供 pick_best_detection 使用
        class TempBoxes:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = xyxy
                self.conf = conf
                self.cls = cls
        temp_boxes = TempBoxes(xyxy_list, conf_list, cls_list)
        result = pick_best_detection(g_frame, temp_boxes, valid)
        if result:
            cropped, label, conf = result
            save_snapshot(cropped, label, conf)
            snap_flash = 12
        else:
            print("  [拍照] 未检测到目标")


def do_exit():
    global running
    running = False


def do_hide_to_tray():
    """Internal helper."""
    global show_desktop, _buttons_built_for_size
    if tray_icon is None:
        print("Status update")
        return
    show_desktop = False
    _buttons_built_for_size = None
    print("Status update")


# ===================== 系统托盘 / 后台隐藏运行 =====================
show_desktop = not (_os.environ.get("YOLO_START_HIDDEN", "0").lower() in {"1", "true", "yes", "on"})
tray_icon = None
window_created = False         # 桌面窗口是否已创建（隐藏时销毁窗口，避免“未响应”）


def _find_hwnd():
    try:
        import win32gui
        return win32gui.FindWindowW(None, win)
    except Exception:
        return None


def _apply_tool_style():
    """把当前 OpenCV 窗口标记为工具窗口（不在任务栏出现）。"""
    try:
        import win32gui, win32con
    except Exception:
        return
    hwnd = _find_hwnd()
    if not hwnd:
        return
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    style |= win32con.WS_EX_TOOLWINDOW
    style &= ~win32con.WS_EX_APPWINDOW
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)


def _create_desktop_window(fullscreen=False):
    """在主线程创建并显示桌面窗口（工具窗口样式，不占任务栏）。"""
    global window_created
    if fullscreen:
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, mouse_callback)
    _apply_tool_style()
    try:
        import win32gui, win32con
        hwnd = _find_hwnd()
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
    except Exception:
        pass
    window_created = True


def _destroy_desktop_window():
    """在主线程销毁桌面窗口，避免隐藏的 HWND 因无人泵消息被判“未响应”。"""
    global window_created
    try:
        cv2.destroyWindow(win)
        cv2.waitKey(1)   # 让销毁事件立即生效
    except Exception:
        pass
    window_created = False


def _enforce_window_size():
    """Internal helper."""
    try:
        rect = cv2.getWindowImageRect(win)
    except Exception:
        return
    if not rect or len(rect) < 4:
        return
    x, y, cw, ch = rect
    if cw <= 0 or ch <= 0:
        return
    new_w, new_h = cw, ch
    # 1) 鍏堟弧瓒虫渶灏忓昂瀵哥害鏉?
    if new_w < MIN_WIN_W:
        new_w = MIN_WIN_W
    if new_h < MIN_WIN_H:
        new_h = MIN_WIN_H
    # 2) 鍐嶉攣瀹?16:9锛氫互瀹藉害涓哄熀鍑嗘帹绠楅珮搴︼紝楂樺害涓嶈冻鏃跺弽鍚戜互楂樺害涓哄噯
    if abs(new_w / max(new_h, 1) - DISPLAY_ASPECT) > 0.01:
        cand_h = round(new_w / DISPLAY_ASPECT)
        if cand_h >= MIN_WIN_H:
            new_h = cand_h
        else:
            new_h = MIN_WIN_H
            new_w = round(new_h * DISPLAY_ASPECT)
    # 浠呭湪鍋忕瀹瑰樊鏃惰皟鐢?resizeWindow锛岄伩鍏嶆瘡甯ф姈鍔ㄦ垨姝诲惊鐜?
    if abs(new_w - cw) > WIN_RESIZE_TOL or abs(new_h - ch) > WIN_RESIZE_TOL:
        try:
            cv2.resizeWindow(win, new_w, new_h)
        except Exception:
            pass


def _tray_show(icon, item):
    # 仅设标志；实际建窗在主线程主循环里完成，避免跨线程操作 OpenCV GUI
    global show_desktop
    show_desktop = True


def _tray_hide(icon, item):
    # 仅设标志；实际销毁窗口在主线程主循环里完成
    global show_desktop
    show_desktop = False


def _tray_exit(icon, item):
    global running
    running = False


def _make_tray_image():
    from PIL import Image, ImageDraw
    s = 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([(6, 6), (58, 58)], fill=(28, 120, 220, 255))
    d.ellipse([(22, 22), (42, 42)], fill=(255, 255, 255, 255))
    d.ellipse([(27, 27), (37, 37)], fill=(28, 120, 220, 255))
    return img


import subprocess as _subprocess
import base64 as _base64

_CLIP_NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW


def _copy_to_clipboard(text):
    """把文本复制到系统剪贴板（Windows）。优先 PowerShell，失败退回 ctypes。
    成功返回 True，失败返回 False。"""
    if not text:
        return False
    # 1) PowerShell Set-Clipboard（对 pythonw 无窗口进程最可靠，支持 Unicode）
    try:
        b64 = _base64.b64encode(text.encode("utf-16-le")).decode("ascii")
        ps = ("Set-Clipboard -Value "
              "([Text.Encoding]::Unicode.GetString("
              "[Convert]::FromBase64String('%s')))" % b64)
        r = _subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            creationflags=_CLIP_NO_WINDOW,
            stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL, timeout=15)
        if r.returncode == 0:
            return True
    except Exception:
        pass
    # 2) 退回 ctypes 原生实现
    try:
        import ctypes
        CF_UNICODETEXT = 13
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.OpenClipboard(0):
            return False
        try:
            user32.EmptyClipboard()
            hMem = kernel32.GlobalAlloc(0x0042, (len(text) + 1) * 2)
            if not hMem:
                return False
            pMem = kernel32.GlobalLock(hMem)
            ctypes.memmove(pMem, text.encode("utf-16-le"), len(text) * 2)
            ctypes.memmove(pMem + len(text) * 2, b"\x00\x00", 2)
            kernel32.GlobalUnlock(hMem)
            user32.SetClipboardData(CF_UNICODETEXT, hMem)
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False
    return False


def _tray_copy_url(icon, item):
    """复制当前观看地址到剪贴板（公网优先，其次局域网），并弹出气泡提示。"""
    url = remote_url or streamer.url
    if _copy_to_clipboard(url):
        msg = "已复制观看地址：\n" + url
    else:
        msg = "复制失败，请手动复制：\n" + url
    try:
        icon.notify(msg, "YOLO 人物识别")
    except Exception:
        pass


def _tray_select_cam(idx):
    """Internal helper."""
    if multi_cam or camera_switching:
        return
    if idx == cur:
        return
    with pending_actions_lock:
        # 鍚屼竴鎽勫儚澶村凡鍦ㄩ槦鍒楁湯灏惧垯涓嶅啀閲嶅鍏ラ槦
        for a in pending_main_actions:
            if isinstance(a, tuple) and a[0] == "select_cam" and a[1] == idx:
                return
        pending_main_actions.append(("select_cam", idx))


def _tray_refresh_cameras(icon, item):
    """Internal helper."""
    do_refresh_cameras()


def _update_tray_menu():
    """Internal helper."""
    if tray_icon is None:
        return
    try:
        tray_icon.menu = build_tray_menu()
    except Exception as e:
        print(f"[托盘] 更新菜单失败: {e}")


def _make_cam_select_action(idx):
    """Internal helper."""
    def _act(icon, item):
        _tray_select_cam(idx)
    return _act


def build_tray_menu():
    """Internal helper."""
    import pystray
    from pystray import Menu, MenuItem
    items = [
        MenuItem("显示画面", _tray_show),
        MenuItem("隐藏画面", _tray_hide),
        MenuItem("\u5237\u65b0\u6444\u50cf\u5934\u5217\u8868", _tray_refresh_cameras),
    ]
    cam_items = []
    for device in cam_devices:
        idx = device['index']
        cam_items.append(
            MenuItem(
                device['label'],
                _make_cam_select_action(idx),
                checked=lambda item, idx=idx: idx == cur,
                radio=True,
            )
        )
    if cam_items:
        items.append(MenuItem("\u5207\u6362\u6444\u50cf\u5934", Menu(*cam_items)))
    items.append(Menu.SEPARATOR)
    items.append(MenuItem("复制观看地址", _tray_copy_url))
    items.append(MenuItem("\u9000\u51fa", _tray_exit))
    return Menu(*items)


def setup_tray():
    global tray_icon
    try:
        import pystray
        from pystray import Menu, MenuItem
        tray_icon = pystray.Icon("yolo_cam_tray", _make_tray_image(),
                                 "YOLO 人物识别", build_tray_menu())
        import threading as _th
        _th.Thread(target=tray_icon.run, daemon=True).start()
        print("Status update")
    except Exception as e:
        global show_desktop
        show_desktop = True   # 托盘不可用则退回可见窗口，避免程序隐形且无法退出
        print(f"Tray startup failed: {e}")


def do_prev_page():
    return


def do_next_page():
    return


def get_items_per_page():
    return 1


def do_quick_focus(idx):
    set_focus(PERSON_CN)


def set_focus(name):
    global focus, focus_cn, focus_id, inp, sel
    focus = True
    focus_cn = PERSON_CN
    focus_id = find_cid(PERSON_CN)
    print(f"-> {PERSON_CN}")
    inp = ""
    sel = False


def build_buttons(w, h):
    """Build clickable buttons based on current screen size and mode."""
    global _buttons_built_for_size, _buttons_cache

    # 缓存按钮：只有当窗口大小或模式改变时才重建
    cache_key = (w, h, sel, focus, auto_snap, snap_mode, full, paused, cur,
                 len(cam_devices), recording, detection_enabled, camera_scan_running,
                 multi_cam, camera_switching, pose_enabled, pose_model_loading,
                 pose_model_ready)
    if _buttons_built_for_size == cache_key and _buttons_cache is not None:
        btn_mgr.buttons = _buttons_cache.copy()
        return

    btn_mgr.clear()
    _buttons_built_for_size = cache_key

    # ====== Normal mode ======
    current_device = next((device for device in cam_devices if device['index'] == cur), None)
    cam_display = current_device['label'] if current_device else f'\u6444\u50cf\u5934{cur}'
    btn_w = 92
    btn_h = 34
    gap = 8
    start_x = 10
    y = h - btn_h - 8

    # Quick focus buttons (bottom row, left side) - 多摄模式下隐藏
    if not multi_cam:
        for i, item in enumerate(quick_items[:1]):
            is_focused = (focus and focus_cn == item)
            bg = (60, 140, 60) if is_focused else (60, 60, 60)
            btn_mgr.add(Button(
                start_x + i * (btn_w + gap), y, btn_w, btn_h,
                f"{i+1}:{item}", lambda idx=i: do_quick_focus(idx),
                color=bg, text_color=(255, 255, 255), font_size=13
            ))

    # Camera picker cards (above quick buttons, left side) - 多摄模式下隐藏
    if not multi_cam:
        cam_card_w = 150
        cam_card_h = 42
        cam_gap = 8
        cam_header_h = 30
        cam_refresh_w = 82
        cam_base_y = y - btn_h - gap

        btn_mgr.add(Button(
            start_x, cam_base_y, cam_card_w * 2 + cam_gap - cam_refresh_w - cam_gap, cam_header_h,
            f"当前: {cam_display}", lambda: None,
            color=(36, 62, 108), text_color=(255, 255, 255), font_size=11
        ))
        btn_mgr.add(Button(
            start_x + cam_card_w * 2 + cam_gap - cam_refresh_w, cam_base_y, cam_refresh_w, cam_header_h,
            "\u626b\u63cf\u4e2d" if camera_scan_running else "\u5237\u65b0\u5217\u8868", do_refresh_cameras,
            color=(88, 108, 138), text_color=(255, 255, 255), font_size=11
        ))

        for i, device in enumerate(cam_devices[:8]):
            row = i // 2
            col = i % 2
            btn_x = start_x + col * (cam_card_w + cam_gap)
            btn_y = cam_base_y - (row + 1) * (cam_card_h + cam_gap)
            is_current = device['index'] == cur
            label = f"[{device['index']}] {device['label']}"
            btn_mgr.add(Button(
                btn_x, btn_y, cam_card_w, cam_card_h,
                label, lambda idx=device['index']: do_select_cam(idx),
                color=(52, 138, 90) if is_current else (40, 80, 140),
                text_color=(255, 255, 255), font_size=11
            ))

    # Function buttons on the right side
    fn_w = 92
    # 澶氭憚妯″紡锛氬彸涓婅鏄簩缁寸爜鎺у埗鏍忥紙x: w-112~w锛寉: 54~208锛夈€傛寜閽粛璐存渶鍙筹紝
    # 浣嗘暣鍒楁斁鍒颁簩缁寸爜鈥滀笅鏂光€濃€斺€旂缉灏忔寜閽珮搴︿笌闂磋窛锛屼娇椤剁浣庝簬 y鈮?08锛?
    # 鏃笉鎸″乏渚ф憚鍍忓ご鐢婚潰锛堢敾闈㈠湪 x<w-112锛夛紝涔熶笉琚悗缁樺埗鐨勪簩缁寸爜鐩栦綇銆?
    if multi_cam:
        fn_h, fn_gap = 30, 6
        fn_x = w - fn_w - 10
    else:
        fn_h, fn_gap = btn_h, gap
        fn_x = w - fn_w - 10
    fn_start_y = h - fn_h - 8
    fn_labels = [
        ("\u5207\u5230\u76d1\u63a7" if detection_enabled else "\u5f00\u542f\u8bc6\u522b", do_toggle_detection, not detection_enabled, (46, 132, 104), False),
        ("\u59ff\u6001\u52a0\u8f7d\u4e2d" if pose_model_loading else ("\u5173\u95ed\u59ff\u6001" if pose_enabled else "\u59ff\u6001\u8bc6\u522b"), do_toggle_pose, pose_enabled, (40, 126, 164), False),
        ("\u62cd\u7167", do_manual_snap, True, (56, 132, 92), False),
        ("\u624b\u673a\u6295\u5c4f", do_toggle_stream_qr, show_stream_qr, (48, 118, 148), False),
        ("\u5f55\u5236" if not recording else "\u505c\u6b62\u5f55\u5236", toggle_recording, recording, (168, 64, 64), False),
        ("\u591a\u6444", do_toggle_multi_cam, multi_cam, (128, 80, 168), False),
        ("\u81ea\u52a8\u62cd", do_toggle_auto_snap, auto_snap, (148, 118, 48), False),
        ("\u5168\u5c4f\u62cd" if snap_mode == "full" else "\u4eba\u7269\u62cd", do_toggle_snap_mode, snap_mode == "full", (88, 96, 148), False),
        ("\u7a97\u53e3\u5316" if full else "\u5168\u5c4f", toggle_full, full, (76, 76, 76), False),
        ("\u7ee7\u7eed" if paused else "\u6682\u505c", do_toggle_pause, paused, (76, 76, 76), False),
        ("\u5e2e\u52a9", do_toggle_help, show_help, (76, 76, 76), False),
        ("\u9690\u85cf\u5230\u6258\u76d8", do_hide_to_tray, False, (76, 76, 76), False),
        ("\u9000\u51fa", do_exit, False, (150, 58, 58), True),
    ]
    for i, (label, cb, is_active, color, is_danger) in enumerate(fn_labels):
        base_color = (156, 62, 62) if is_danger else ((52, 138, 90) if is_active else (72, 72, 72))
        btn_mgr.add(Button(
            fn_x, fn_start_y - (i + 1) * (fn_h + fn_gap), fn_w, fn_h,
            label, cb,
            color=base_color,
            font_size=12
        ))

    _buttons_cache = btn_mgr.buttons.copy()


def mouse_callback(event, mx, my, flags, param):
    global full, show_help

    if event == cv2.EVENT_LBUTTONDBLCLK:
        toggle_full()
        return

    if event == cv2.EVENT_MOUSEMOVE:
        btn_mgr.update_hover(mx, my)
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_mgr.handle_click(mx, my):
            return


# 默认后台隐藏运行：不创建任何 OpenCV 窗口（无 HWND 就不会被判“未响应”）。
# 需要显示时（托盘“显示画面”）由主循环在主线程创建窗口。
if show_desktop:
    _create_desktop_window(full)

# 启动系统托盘（后台隐藏运行）；失败则退回普通窗口
setup_tray()
# Start expensive background initialization only after the first camera frame has
# completed its display path. This guarantees that the user never stares at a
# blank OpenCV window while the GPU model or extra cameras are being prepared.
startup_jobs_started = False

print("\n===== Controls =====")
print("  Q:Quit  F:Fullscreen  C/X:Backup camera switch  P:Pause")
print("  Click left camera cards to choose a working camera")
print("  Use the refresh button to rescan camera devices")
print("  1:Lock person  H:Help  Space:Snapshot  R:Record  B:QR target")
print("  M:Snapshot mode  V:Toggle phone panel  D:Detection/Monitor  K:Pose")
print("====================\n")

# 瀹炴椂甯х巼杩借釜
fps_frame_count = 0
fps_start_time = time.time()
fps_realtime = 0.0
last_overlay_second = ""
last_overlay_text_w = 0

running = True
while running:
    while True:
        with pending_actions_lock:
            _main_action = pending_main_actions.popleft() if pending_main_actions else None
        if _main_action is None:
            break
        if _main_action == "multi_toggle":
            do_toggle_multi_cam()
        elif _main_action == "pose_toggle":
            do_toggle_pose()
        elif isinstance(_main_action, tuple) and _main_action[0] == "select_cam":
            do_select_cam(_main_action[1])
            _update_tray_menu()   # 鍒囧畬鍚庡埛鏂版墭鐩樿彍鍗曠殑褰撳墠鎽勫儚澶村嬀閫?
    phone_frame_offered = False  # reset once per display loop
    # ====== 多摄像头平铺模式（跳过单摄读取） ======
    phone_canvas = None   # 手机端专用画面（默认 None，下面分支会赋值）
    if multi_cam:
        phone_canvas = None   # 手机端专用画面（干净、无桌面 UI），多摄时上下竖排
        phone_stack = []      # 澶氭憚鍚勮矾鐨勫共鍑€鐢婚潰锛岀珫鎺掓嫾鎺ョ敤
        pose_people_count = 0
        pose_visible_keypoints = 0
        detected_people_count = 0
        multi_frame_counter += 1
        multi_infer_interval = 4 if pose_enabled else MULTI_TILE_INFER_N
        multi_do_infer = detection_enabled and active_model_ready() and (multi_frame_counter % multi_infer_interval == 0)
        raw_frames = read_all_frames()
        if multi_do_infer and raw_frames:
            try:
                # Batch all cameras into one GPU call. This avoids per-camera
                # framework overhead and keeps remote pose video visibly smoother.
                inference_model = active_inference_model()
                batch_results = inference_model.predict(
                    raw_frames,
                    imgsz=MULTI_POSE_INFER_SIZE if pose_enabled else MULTI_INFER_SIZE,
                    verbose=False,
                    half=USE_HALF,
                    classes=[0 if pose_enabled else focus_id],
                    conf=PERSON_CONF,
                    iou=PERSON_IOU,
                )
                for _ri, _result in enumerate(batch_results[:len(multi_cached_results)]):
                    multi_cached_results[_ri] = result_to_cache(_result)
            except Exception as exc:
                multi_cached_results = [empty_inference_cache() for _ in multi_cached_results]
                if multi_frame_counter % 60 == 0:
                    print(f"Multi-camera batch inference failed: {exc}")
        n_valid = sum(1 for th in multi_threads if not th.error)
        rows, cols = calc_grid(n_valid) if n_valid else (1, 1)
        grid_w = OUTPUT_W - MULTI_CONTROL_W
        grid_h = OUTPUT_H - MULTI_TOP_BAR_H
        canvas = np.full((OUTPUT_H, OUTPUT_W, 3), (16, 18, 22), dtype=np.uint8)
        cv2.line(canvas, (grid_w, MULTI_TOP_BAR_H), (grid_w, OUTPUT_H), (54, 58, 66), 1)

        # 浠呮覆鏌撴湁淇″彿鐨勬憚鍍忓ご锛涙棤淇″彿鐨勮矾鐩存帴璺宠繃锛岀敾闈㈣嚜閫傚簲閲嶆帓锛堜笉鏄剧ず榛戞牸锛夈€?
        valid_idx = [i for i, th in enumerate(multi_threads) if not th.error]
        valid_threads = [multi_threads[i] for i in valid_idx]
        valid_frames = [raw_frames[i] for i in valid_idx]
        valid_cached = [multi_cached_results[i] for i in valid_idx]

        for vi, th in enumerate(valid_threads):
            r = vi // cols
            c = vi % cols
            tx = c * grid_w // cols
            ty = MULTI_TOP_BAR_H + r * grid_h // rows
            tile_x2 = (c + 1) * grid_w // cols
            tile_y2 = MULTI_TOP_BAR_H + (r + 1) * grid_h // rows
            tile_w = tile_x2 - tx
            tile_h = tile_y2 - ty
            content_y = ty + MULTI_TILE_INFO_H
            content_h = max(1, tile_h - MULTI_TILE_INFO_H)
            tile_frame = valid_frames[vi]
            label = th.label
            # 保持宽高比缩放 + 黑边填充（letterbox）
            fh, fw = tile_frame.shape[:2]
            tile_area, scale, ox, oy, new_w, new_h = fit_complete_on_blur(
                tile_frame, tile_w, content_h)
            canvas[content_y:tile_y2, tx:tile_x2] = tile_area

            # 绘制检测框（框在原始 tile 坐标，按显示缩放+偏移映射到画布）
            res = valid_cached[vi]
            detected_people_count += len(res.get('xyxy', []))
            if pose_enabled:
                _pose_people, _pose_points = get_pose_metrics(res)
                pose_people_count += _pose_people
                pose_visible_keypoints += _pose_points
            # 每路摄像头使用自己的原始画面、检测结果和冷却状态自动抓拍。
            update_position_memory(tile_frame, res, ("camera", th.idx))
            maybe_auto_snapshot(tile_frame, res, ('camera', th.idx), f'\u6444\u50cf\u5934{th.idx}')
            sx = new_w / max(fw, 1)
            sy = new_h / max(fh, 1)
            if 'xyxy' in res and res['xyxy'] is not None and len(res['xyxy']) > 0:
                for bi in range(len(res['xyxy'])):
                    x1, y1, x2, y2 = map(int, res['xyxy'][bi])
                    conf = float(res['conf'][bi]) if len(res['conf']) > bi else 0
                    x1 = int(x1 * sx) + tx + ox
                    y1 = int(y1 * sy) + content_y + oy
                    x2 = int(x2 * sx) + tx + ox
                    y2 = int(y2 * sy) + content_y + oy
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 120), 2)
                    label_y = max(content_y + 16, y1 - 6)
                    canvas = draw_cn(canvas, f"人 {conf:.2f}", (x1 + 4, label_y), 12, (0, 0, 0), (0, 220, 120))
            if pose_enabled:
                def _desktop_pose_transform(px, py, _sx=sx, _sy=sy,
                                            _tx=tx + ox, _ty=content_y + oy):
                    return px * _sx + _tx, py * _sy + _ty
                draw_pose_skeleton(canvas, res.get('keypoints'),
                                   res.get('keypoint_conf'), _desktop_pose_transform)

            # 手机端专用干净画面：仅摄像头+检测框，不带桌面 UI（按钮/标题/信息面板）。
            # 多摄时改为上下竖排，避免平铺导致每路画面过小、看不清。
            _pt = tile_frame.copy()
            if 'xyxy' in res and res['xyxy'] is not None and len(res['xyxy']) > 0:
                for bi in range(len(res['xyxy'])):
                    _bx1, _by1, _bx2, _by2 = map(int, res['xyxy'][bi])
                    cv2.rectangle(_pt, (_bx1, _by1), (_bx2, _by2), (0, 220, 120), 2)
                    _conf = float(res['conf'][bi]) if len(res['conf']) > bi else 0
                    _pt = draw_cn(_pt, f"人 {_conf:.2f}", (_bx1 + 4, _by1 - 6), 12, (0, 0, 0), (0, 220, 120))
            if pose_enabled:
                draw_pose_skeleton(_pt, res.get('keypoints'), res.get('keypoint_conf'))
            _ps = PHONE_STACK_W / max(fw, 1)
            _pt = cv2.resize(_pt, (PHONE_STACK_W, int(fh * _ps)), interpolation=cv2.INTER_LINEAR)
            phone_stack.append(_pt)

            # 摄像头信息：两行独立栏，名称/分辨率在左，时间/帧率在右。
            cv2.rectangle(canvas, (tx, ty), (tile_x2 - 1, content_y - 1), (24, 28, 34), -1)
            cv2.line(canvas, (tx, content_y - 1), (tile_x2 - 1, content_y - 1), (72, 78, 88), 1)
            multi_now = datetime.datetime.now()
            time_str = multi_now.strftime("%H:%M:%S")
            time_w = text_pixel_width(time_str, 12)
            label_max_w = max(20, tile_w - time_w - 26)
            label_txt = ellipsize_text(label, label_max_w, 13)
            canvas = draw_cn(canvas, label_txt, (tx + 8, ty + 3), 13, (90, 245, 150))
            canvas = draw_cn(canvas, time_str, (tile_x2 - 8, ty + 3), 12,
                             (235, 235, 215), anchor="rt")
            cap_w_ = int(th.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or fw
            cap_h_ = int(th.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fh
            res_str = f"{cap_w_}x{cap_h_}"
            fps_val = th.fps
            status_str = "\u65e0\u4fe1\u53f7" if th.error else f"{fps_val:.1f} FPS"
            status_color = (120, 160, 255) if th.error else (90, 230, 255)
            canvas = draw_cn(canvas, res_str, (tx + 8, ty + 24), 11, (190, 196, 205))
            canvas = draw_cn(canvas, status_str, (tile_x2 - 8, ty + 24), 11,
                             status_color, anchor="rt")

        # 手机端多摄画面（干净、无桌面 UI）
        if streamer.phone_layout == "tile":
            # 横屏/全屏：平铺，每路占满宽，黑边更少
            _rows, _cols = calc_grid(n_valid) if n_valid else (1, 1)
            _tw, _th = 1280 // _cols, 720 // _rows
            pt_canvas = np.zeros((720, 1280, 3), np.uint8)
            for vi, th_ in enumerate(valid_threads):
                r = vi // _cols
                c = vi % _cols
                _tx = c * _tw
                _ty = r * _th
                _f = valid_frames[vi]
                fh, fw = _f.shape[:2]
                _tile, _s, _ox, _oy, _nw, _nh = fit_complete_on_blur(_f, _tw, _th)
                pt_canvas[_ty:_ty + _th, _tx:_tx + _tw] = _tile
                _res = valid_cached[vi]
                if 'xyxy' in _res and _res['xyxy'] is not None and len(_res['xyxy']) > 0:
                    _sx, _sy = _nw / max(fw, 1), _nh / max(fh, 1)
                    for bi in range(len(_res['xyxy'])):
                        x1, y1, x2, y2 = map(int, _res['xyxy'][bi])
                        x1 = int(x1 * _sx) + _tx + _ox
                        y1 = int(y1 * _sy) + _ty + _oy
                        x2 = int(x2 * _sx) + _tx + _ox
                        y2 = int(y2 * _sy) + _ty + _oy
                        cv2.rectangle(pt_canvas, (x1, y1), (x2, y2), (0, 220, 120), 2)
                if pose_enabled:
                    def _phone_tile_pose_transform(px, py, __sx=_nw / max(fw, 1),
                                                   __sy=_nh / max(fh, 1),
                                                   __tx=_tx + _ox, __ty=_ty + _oy):
                        return px * __sx + __tx, py * __sy + __ty
                    draw_pose_skeleton(pt_canvas, _res.get('keypoints'),
                                       _res.get('keypoint_conf'), _phone_tile_pose_transform)
            phone_canvas = pt_canvas if n_valid else make_no_signal_frame()
        else:
            # 竖屏：上下竖排拼接（无桌面 UI）
            if phone_stack:
                phone_canvas = np.vstack(phone_stack)
            else:
                phone_canvas = make_no_signal_frame()

        if phone_canvas is None:
            phone_canvas = make_no_signal_frame()
        if phone_canvas is not None and streamer.client_count() > 0:
            streamer.update(phone_canvas)
            phone_frame_offered = True

        g_frame = canvas.copy()
        g_boxes = None
        ann = canvas
        # 进入多摄：多摄网格交叉淡入（旧单摄画面缓出）
        if cam_transition['active'] and cam_transition['old'] is not None:
            _p = (time.time() - cam_transition['t0']) / cam_transition['dur']
            if _p >= 1.0:
                cam_transition['active'] = False
            else:
                _e = _p * _p * (3 - 2 * _p)
                ann = cv2.addWeighted(canvas, _e, cam_transition['old'], 1 - _e, 0)
        frame_h, frame_w = OUTPUT_H, OUTPUT_W
        cam_w, cam_h = OUTPUT_W, OUTPUT_H

        # 鍏ㄥ眬 FPS锛堢敤浜庢棩鏈熶笅鏂规樉绀猴級
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_realtime = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
    elif not paused:
        ret, frame = cap.read()
        if not ret:
            # 鎽勫儚澶存柇娴侊細閲婃斁骞跺皾璇曢噸杩烇紝涓嶉€€鍑虹▼搴?
            try:
                cap.release()
            except Exception:
                pass
            cap, cam_w, cam_h, cam_fps = open_cam(cur)
            ret, frame = cap.read()
        if not ret:
            # 浠嶆棤淇″彿锛氬悜鎵嬫満绔帹閫佸崰浣嶅抚锛屽悗鍙版寔缁噸璇曢噸杩烇紙涓嶅啀閫€鍑虹▼搴忥級
            if streamer.client_count() > 0:
                streamer.update(make_no_signal_frame())
                phone_frame_offered = True
            time.sleep(0.5)
            continue

        if frame.shape[1] != OUTPUT_W or frame.shape[0] != OUTPUT_H:
            frame = cv2.resize(frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        cam_w, cam_h = OUTPUT_W, OUTPUT_H

        # Pure monitoring fast path: send the raw camera frame immediately,
        # before inference bookkeeping and desktop UI rendering.
        if not detection_enabled and streamer.client_count() > 0:
            phone_canvas = frame
            streamer.update(phone_canvas)
            phone_frame_offered = True

        frame_counter += 1
        if pose_enabled:
            # Newest-frame asynchronous pose inference: video capture and MJPEG
            # delivery never wait for the keypoint network.
            if detection_enabled and active_model_ready() and frame_counter % POSE_INFER_EVERY_N == 0:
                pose_async.submit(frame)
            boxes = pose_async.latest()
            cached_boxes = boxes
            if pose_async.completed and not _first_inference_logged:
                print(f"First async pose inference finished in {pose_async.inference_ms:.1f} ms.")
                _first_inference_logged = True
        else:
            do_infer = detection_enabled and active_model_ready() and (frame_counter % INFER_EVERY_N == 0)
            if do_infer:
                if not _first_inference_logged:
                    print("First live inference started.")
                _infer_t0 = time.perf_counter()
                results = model.track(
                    frame, imgsz=INFER_SIZE, verbose=False, half=USE_HALF,
                    persist=True, tracker="bytetrack.yaml", classes=[focus_id],
                    conf=PERSON_CONF, iou=PERSON_IOU,
                )
                if not _first_inference_logged:
                    print(f"First detect inference finished in {(time.perf_counter() - _infer_t0) * 1000:.1f} ms.")
                    _first_inference_logged = True
                cached_boxes = result_to_cache(results[0])
            boxes = cached_boxes

        if pose_enabled:
            pose_people_count, pose_visible_keypoints = get_pose_metrics(boxes)
        else:
            pose_people_count = 0
            pose_visible_keypoints = 0

        g_frame = frame.copy()
        g_boxes = boxes
        ann = frame.copy()
        # 摄像头切换：新画面交叉淡入（旧画面缓出），消除硬切卡顿感
        if cam_transition['active'] and cam_transition['old'] is not None:
            _p = (time.time() - cam_transition['t0']) / cam_transition['dur']
            if _p >= 1.0:
                cam_transition['active'] = False
            else:
                _e = _p * _p * (3 - 2 * _p)   # smoothstep 缂撳姩
                ann = cv2.addWeighted(frame, _e, cam_transition['old'], 1 - _e, 0)
        frame_h, frame_w = frame.shape[:2]

        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_realtime = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        xyxy_list, conf_list, cls_list, track_id_list = get_boxes_data(boxes)
        candidates = []
        for i in range(len(xyxy_list)):
            x1, y1, x2, y2 = map(int, xyxy_list[i])
            if not is_valid_person_box(x1, y1, x2, y2):
                continue
            conf = conf_list[i]
            cid = int(cls_list[i])
            if focus and cid != focus_id:
                continue
            track_id = int(track_id_list[i]) if i < len(track_id_list) else -1
            area = max(0, (x2 - x1) * (y2 - y1))
            candidates.append({
                'track_id': track_id,
                'conf': conf,
                'cid': cid,
                'box': (x1, y1, x2, y2),
                'area': area,
            })

        # 姣忓抚鏇存柊浣嶇疆鍗犵敤璁板繂锛堝骇浣嶈蹇嗭級锛屼笌鑷姩鎷嶇収寮€鍏虫棤鍏?
        detected_people_count = len(candidates)
        update_position_memory(frame, boxes, ("camera", cur))

        active_target = None
        if locked_track_id is not None:
            for candidate in candidates:
                if candidate['track_id'] == locked_track_id:
                    active_target = candidate
                    locked_misses = 0
                    break
            if active_target is None:
                locked_misses += 1
                if locked_misses > LOCK_MISS_TOLERANCE:
                    locked_track_id = None
                    locked_box = None
                    locked_conf = 0.0
                    locked_misses = 0

        if active_target is None and candidates:
            tracked = [c for c in candidates if c['track_id'] >= 0]
            pool = tracked if tracked else candidates
            active_target = max(pool, key=lambda c: c['conf'] + c['area'] / max(frame_w * frame_h, 1))
            if active_target['track_id'] >= 0:
                locked_track_id = active_target['track_id']
                locked_misses = 0
            locked_box = active_target['box']
            locked_conf = active_target['conf']

        if active_target is not None:
            locked_box = active_target['box']
            locked_conf = active_target['conf']

        for candidate in candidates:
            x1, y1, x2, y2 = candidate['box']
            cid = candidate['cid']
            conf = candidate['conf']
            track_id = candidate['track_id']
            is_locked = locked_track_id is not None and track_id == locked_track_id
            color = (0, 220, 120) if is_locked else (96, 96, 96)
            thickness = 3 if is_locked else 1
            cn_label = get_cn(all_names[cid])
            track_txt = f" #{track_id}" if track_id >= 0 else ""
            label = f"{cn_label}{track_txt} {conf:.2f}"
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, thickness)
            ann = draw_cn(ann, label, (x1 + 2, y1 - 24), 16,
                          (255, 255, 255) if is_locked else (210, 210, 210), color)

        if pose_enabled and isinstance(boxes, dict):
            draw_pose_skeleton(ann, boxes.get('keypoints'), boxes.get('keypoint_conf'))

        # 手机端专用画面：竖屏自适应——摄像头居中，上下留白区显示识别统计。
        # 仅在有手机观看时才合成，避免无人观看时白白消耗 CPU。
        if not phone_frame_offered:
            if streamer.client_count() > 0:
                # Send the original 16:9 annotated frame. The browser uses
                # object-fit: contain to adapt it to phones, tablets and desktop
                # screens without server-side portrait padding or cropping.
                phone_canvas = ann
                streamer.update(phone_canvas)
                phone_frame_offered = True
            else:
                phone_canvas = ann

        if locked_box is not None and locked_track_id is not None:
            lx1, ly1, lx2, ly2 = locked_box
            ann = draw_cn(ann, f"[锁定人物 #{locked_track_id}]", (10, 52), 15,
                          (255, 255, 255), (0, 150, 90))
    else:
        # Paused: still need a valid ann frame for display
        pass

    h, w = ann.shape[:2]

    if sel:
        sel = False

    if show_desktop:
        # ====== Normal detection mode ======
        if not detection_enabled:
            title_txt = "[\u7eaf\u76d1\u63a7\u6a21\u5f0f] \u5df2\u5173\u95ed\u6a21\u578b\u8bc6\u522b, \u6309D\u6062\u590d"
        elif pose_enabled and pose_model_loading:
            title_txt = "[\u59ff\u6001\u6a21\u578b\u52a0\u8f7d\u4e2d] \u753b\u9762\u4fdd\u6301\u6d41\u7545"
        elif pose_enabled and pose_load_error and not pose_model_ready:
            title_txt = "[\u59ff\u6001\u6a21\u578b\u52a0\u8f7d\u5931\u8d25]"
        elif not active_model_ready():
            title_txt = "[\u6a21\u578b\u52a0\u8f7d\u4e2d] \u6444\u50cf\u5934\u753b\u9762\u5df2\u663e\u793a"
        else:
            title_txt = f"[{current_recognition_name()}] K:\u5207\u6362\u59ff\u6001 M:\u62cd\u7167\u6a21\u5f0f"

        if multi_cam:
            # 多摄全局状态集中在独立顶部栏，不覆盖第一排摄像头信息。
            cv2.rectangle(ann, (0, 0), (w - MULTI_CONTROL_W - 1, MULTI_TOP_BAR_H - 1),
                          (18, 22, 28), -1)
            cv2.line(ann, (0, MULTI_TOP_BAR_H - 1),
                     (w - MULTI_CONTROL_W - 1, MULTI_TOP_BAR_H - 1), (64, 70, 80), 1)
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d %H:%M:%S").replace("-0", "-")
            date_w = text_pixel_width(date_str, 13)
            title_max_w = max(120, w - MULTI_CONTROL_W - date_w - 48)
            title_txt = ellipsize_text(title_txt, title_max_w, 16)
            ann = draw_cn(ann, title_txt, (12, 5), 16, (90, 245, 150))
            ann = draw_cn(ann, date_str, (w - MULTI_CONTROL_W - 12, 6), 13,
                          (235, 235, 215), anchor="rt")

            state_parts = ["\u5df2\u6682\u505c" if paused else "\u8fd0\u884c\u4e2d"]
            if pose_enabled:
                pose_state = "\u59ff\u6001\u52a0\u8f7d\u4e2d" if pose_model_loading else f"\u59ff\u6001:{pose_people_count}\u4eba/{pose_visible_keypoints}\u70b9"
                state_parts.append(pose_state)
            if recording:
                state_parts.append("\u5f55\u5236\u4e2d")
            if auto_snap:
                state_parts.append("\u81ea\u52a8\u62cd\u7167")
            state_txt = "\u72b6\u6001: " + " | ".join(state_parts)
            overview_txt = f"{len(multi_threads)}\u8def\u753b\u9762 | {fps_realtime:.1f} FPS"
            overview_w = text_pixel_width(overview_txt, 12)
            state_max_w = max(80, w - MULTI_CONTROL_W - overview_w - 48)
            state_txt = ellipsize_text(state_txt, state_max_w, 12)
            ann = draw_cn(ann, state_txt, (12, 30), 12,
                          (255, 210, 120) if paused else (190, 196, 205))
            ann = draw_cn(ann, overview_txt, (w - MULTI_CONTROL_W - 12, 30), 12,
                          (90, 230, 255), anchor="rt")
        else:
            if paused:
                ann = draw_cn(ann, "[已暂停] 按P继续", (w // 2 - 100, 30), 24,
                              (0, 255, 255), (0, 0, 0))
            ann = draw_cn(ann, title_txt, (10, 28), 16, (0, 255, 0), (30, 30, 30))
            if recording and not show_stream_qr:
                ann = draw_cn(ann, "\u25cf \u5f55\u5236\u4e2d", (w // 2 - 44, 30), 18,
                              (255, 90, 90), (40, 0, 0))

        # ====== Build and draw buttons ======
        btn_mgr.tick_animations(time.time())
        build_buttons(w, h)
        ann = btn_mgr.draw_all(ann, time.time())

        # Help overlay
        if show_help and not sel:
            help_texts = [
                "===== HELP =====",
                "Q:Quit F:Fullscreen P:Pause",
                "Click camera cards to switch camera",
                "Refresh list after plugging cameras",
                "Space:Snapshot R:Record B:QR",
                "M:Snapshot mode K:Pose D:Monitor",
            ]
            overlay_h = len(help_texts) * 26 + 20
            overlay2 = np.full((overlay_h, w, 3), (0, 0, 0), dtype=np.uint8)
            ann[:overlay_h, :] = cv2.addWeighted(ann[:overlay_h, :], 0.4, overlay2, 0.6, 0)
            for i, txt in enumerate(help_texts):
                ann = draw_cn(ann, txt, (20, 10 + i * 26), 18, (0, 255, 255))

        # ====== Date/time display (top-right, won't block buttons) ======
        if not multi_cam:
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d %H:%M:%S").replace("-0", "-")
            tmp_font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
            if date_str != last_overlay_second:
                tmp_pil = Image.new("RGB", (1, 1))
                tmp_d = ImageDraw.Draw(tmp_pil)
                bbox = tmp_d.textbbox((0, 0), date_str, font=tmp_font)
                last_overlay_text_w = bbox[2] - bbox[0] + 8
                last_overlay_second = date_str
            date_x = w - last_overlay_text_w - 10
            date_y1 = 8
            date_y2 = 26
            # 自动感知背景亮度，选择高对比度文字颜色
            def _contrast_color(img, x, y, text_w, text_h):
                """根据背景区域平均亮度返回 (text_color, bg_color)"""
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img.shape[1], x + text_w + 10)
                y2 = min(img.shape[0], y + text_h + 6)
                region = img[y1:y2, x1:x2]
                if region.size == 0:
                    return (255, 255, 200), None
                avg_brightness = np.mean(region)
                if avg_brightness > 140:
                    return (20, 20, 20), (220, 220, 220)  # 亮背景 → 深色字 + 浅背景
                else:
                    return (255, 255, 200), (30, 30, 30)   # 暗背景 → 浅色字 + 深背景
            date_color, date_bg = _contrast_color(ann, date_x, date_y1 - 2, last_overlay_text_w, 32)
            ann = draw_cn(ann, date_str, (date_x, date_y1), 14, date_color, date_bg)
            # 分辨率和实时帧率信息（在日期下方）
            res_str = f"{cam_w}x{cam_h} @ {fps_realtime:.1f}fps"
            res_color, res_bg = _contrast_color(ann, date_x, date_y2 - 2, last_overlay_text_w, 20)
            ann = draw_cn(ann, res_str, (date_x, date_y2), 14, res_color, res_bg)

    # ====== Auto-snapshot ======
    # 多摄模式已在每个 tile 内分别处理；单摄使用当前原始帧与当前检测结果。
    if not multi_cam:
        maybe_auto_snapshot(frame, boxes, ('camera', cur), f'\u6444\u50cf\u5934{cur}')

    # Flash effect
    if snap_flash > 0:
        k = snap_flash / 12.0                      # 1 鈫?0锛氶殢甯ф暟鑷劧娣″嚭
        alpha = 0.12 * k                           # 宄板€间粎 12% 鐧借挋鐗?
        white = np.full_like(ann, 255)
        ann = cv2.addWeighted(ann, 1.0 - alpha, white, alpha, 0)
        snap_flash -= 1

    # Auto-snap indicator is already included in the multicamera top status bar.
    if auto_snap and not multi_cam:
        mode_txt = f"[自动拍:{'开' if auto_snap else '关'}|{'全屏' if snap_mode=='full' else '人物'}]"
        ann = draw_cn(ann, mode_txt, (10, 50), 15, (0, 200, 255), (0, 0, 0))

    # 搴т綅璁板繂锛堜綅缃崰鐢級鎽樿锛氬崟鎽勬ā寮忓簳閮ㄥ乏渚ф樉绀哄崰鐢ㄦ儏鍐?
    if detection_enabled and not multi_cam:
        _pmap = get_position_map_summary(("camera", cur))
        if _pmap:
            _occ = [p for p in _pmap if p["occupied"]]
            _occ_n = len(_occ)
            _new = [str(p["pos"]) for p in _occ
                    if p["track_id"] is not None and p["dwell_s"] <= 5]
            _detail = " ".join(f"#{p['track_id']}@{p['pos']}({p['dwell_s']}s)" for p in _occ)
            _seat_txt = f"位置记忆 {_occ_n}/{len(_pmap)} 占用"
            if _new:
                _seat_txt += f" | 新来位置: {'/'.join(_new)}"
            ann = draw_cn(ann, _seat_txt, (10, h - 30), 14, (120, 220, 255), (0, 0, 0))
            if _detail:
                ann = draw_cn(ann, _detail, (10, h - 12), 12, (180, 200, 210), (0, 0, 0))


    # ====== 录制（录屏回放）：写入当前画面 ======
    if recording and video_writer is not None:
        with rec_lock:
            if recording and video_writer is not None:
                try:
                    video_writer.write(ann)
                except Exception:
                    pass

    # ====== 推送画面到手机（在叠加二维码之前，手机端不显示二维码本身）======
    # 推流到手机：仅发送干净摄像头画面（无桌面按钮/标题/二维码等 UI）
    if not phone_frame_offered:
        streamer.update(phone_canvas if phone_canvas is not None else ann)

    # ====== 鎵嬫満鎶曞睆锛氱獥鍙ｄ笂鍙犲姞浜岀淮鐮佷笌璁块棶鍦板潃 ======
    if show_stream_qr and show_desktop:
        if multi_cam:
            # 多摄使用右侧控制栏内的紧凑二维码，避免遮住第一路信息栏与 FPS。
            panel_x = w - MULTI_CONTROL_W
            panel_y = MULTI_TOP_BAR_H
            panel_w = MULTI_CONTROL_W
            panel_h = 154
            cv2.rectangle(ann, (panel_x, panel_y),
                          (w - 1, min(h - 1, panel_y + panel_h)), (20, 24, 30), -1)
            qr_size = 82
            qr_x = panel_x + (panel_w - qr_size) // 2
            qr_y = panel_y + 8
            if _qr_img is not None and qr_y + qr_size <= h:
                qr_resized = cv2.resize(_qr_img, (qr_size, qr_size),
                                        interpolation=cv2.INTER_NEAREST)
                ann[qr_y:qr_y + qr_size, qr_x:qr_x + qr_size] = qr_resized
            target_label = "公网二维码" if qr_target == "public" else "局域网二维码"
            label_x = panel_x + max(4, (panel_w - text_pixel_width(target_label, 11)) // 2)
            ann = draw_cn(ann, target_label, (label_x, qr_y + qr_size + 7), 11,
                          (120, 235, 200))
            online_txt = f"在线 {streamer._clients}"
            online_x = panel_x + max(4, (panel_w - text_pixel_width(online_txt, 10)) // 2)
            ann = draw_cn(ann, online_txt, (online_x, qr_y + qr_size + 29), 10,
                          (180, 196, 215))
        else:
            qr_size = 116
            pad = 10
            panel_x, panel_y = 12, 78
            panel_w = qr_size + 2 * pad + 244
            panel_h = 150
            ox2 = min(w, panel_x + panel_w)
            oy2 = min(h, panel_y + panel_h)
            if oy2 > panel_y and ox2 > panel_x:
                sub = ann[panel_y:oy2, panel_x:ox2]
                dark = np.zeros_like(sub)
                ann[panel_y:oy2, panel_x:ox2] = cv2.addWeighted(sub, 0.30, dark, 0.70, 0)
            # 二维码
            if _qr_img is not None:
                qy2 = panel_y + pad + qr_size
                qx2 = panel_x + pad + qr_size
                if qy2 <= h and qx2 <= w:
                    qr_resized = cv2.resize(_qr_img, (qr_size, qr_size),
                                            interpolation=cv2.INTER_NEAREST)
                    ann[panel_y + pad:qy2, panel_x + pad:qx2] = qr_resized
            # 鏂囨湰
            tx = panel_x + pad + qr_size + 12
            ann = draw_cn(ann, "手机同步观看", (tx, panel_y + pad + 2), 16, (0, 255, 180))
            ann = draw_cn(ann, f"局域网: {stream_url}", (tx, panel_y + pad + 30), 13, (255, 230, 120))
            line_pub = f"公网: {remote_url}" if remote_url else "公网: 未连接(按B重试)"
            ann = draw_cn(ann, line_pub, (tx, panel_y + pad + 54), 13,
                          (255, 180, 120) if remote_url else (200, 200, 200))
            target_txt = f"在线: {streamer._clients}  [{'公网' if qr_target == 'public' else '局域网'}二维码]"
            ann = draw_cn(ann, target_txt, (tx, panel_y + pad + 80), 12, (150, 220, 255))
            if recording:
                ann = draw_cn(ann, "\u25cf \u5f55\u5236\u4e2d", (tx, panel_y + pad + 104), 13, (255, 90, 90))

    if show_desktop:
        if not window_created:
            _create_desktop_window(full)
        # 闈炲叏灞忔椂閿佸畾绐楀彛涓?16:9 涓斾笉灏忎簬鏈€灏忓昂瀵革紝閬垮厤鐢婚潰/鎸夐挳琚媺浼?
        if not full:
            _enforce_window_size()
        cv2.imshow(win, ann)
        key = cv2.waitKey(1) & 0xFF
    else:
        if window_created:
            _destroy_desktop_window()
        # 后台隐藏运行：无窗口、无 HWND，不会被判“未响应”；轻量节流
        key = -1
        time.sleep(0.005)

    if not startup_jobs_started:
        startup_jobs_started = True
        if detection_enabled:
            start_model_warm_async()
        do_refresh_cameras()

    # ====== Global key handlers ======
    if key == ord("q"):
        break
    elif key == ord("f"):
        toggle_full()
    elif key == ord("c"):
        do_switch_cam(1)
    elif key == ord("x"):
        do_switch_cam(-1)
    elif key == ord("p"):
        do_toggle_pause()
    elif key == ord("d"):
        do_toggle_detection()
    elif key == ord("k"):
        do_toggle_pose()
    elif key == ord(" "):  # Space - manual snapshot
        do_manual_snap()
    elif key == ord("s"):
        do_toggle_focus()
    elif key == ord("t"):
        do_toggle_selector()
    elif key == ord("h"):
        do_toggle_help()
    elif key == ord("m"):
        do_toggle_snap_mode()
    elif key == ord("v"):
        do_toggle_stream_qr()
    elif key == ord("r"):
        toggle_recording()
    elif key == ord("b"):
        do_toggle_qr_target()
    elif key == 27:  # Esc
        if sel:
            sel = False
            inp = ""
            sel_page = 0
        elif show_help:
            show_help = False

    # ====== Quick key 1 ======
    if not sel and key == ord("1"):
        do_quick_focus(0)

if recording:
    stop_recording()
if 'tunnel' in globals() and tunnel is not None:
    try:
        tunnel.stop()
    except Exception:
        pass
streamer.stop()
close_all_cameras()
if tray_icon is not None:
    try:
        tray_icon.stop()
    except Exception:
        pass
cap.release()
cv2.destroyAllWindows()
print("Stopped.")
