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

# ====== 模型选择：GPU 用精度更高的 yolov8m，CPU 用轻量 yolov8n ======
# 想换模型可改这里的候选顺序（如 yolo26n.pt、yolov8s.pt 等）
if torch.cuda.is_available():
    _MODEL_CANDIDATES = ["yolov8m.pt", "models/yolov8m.pt",
                         "yolov8n.pt", "models/yolov8n.pt"]
else:
    _MODEL_CANDIDATES = ["yolov8n.pt", "models/yolov8n.pt", "yolo26n.pt"]

model_path = None
for _cand in _MODEL_CANDIDATES:
    _p = BASE_DIR / _cand
    if _p.exists():
        model_path = _p
        break
if model_path is None:
    model_path = BASE_DIR / "models" / "yolov8n.pt"

print(f"Loading model: {model_path}")
model = YOLO(str(model_path))
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
PERSON_CONF = 0.50   # 降低阈值以提升识别率（更少漏检）
PERSON_IOU = 0.45
MIN_PERSON_BOX_AREA = 18000
MIN_PERSON_BOX_HEIGHT = 160

# ====== Snapshot (拍照) Settings ======
SNAP_DIR = BASE_DIR / "snapshots"
SNAP_DIR.mkdir(exist_ok=True)
MAX_STORAGE_GB = 10  # Max storage size in GB (adjustable)
snapshot_lock = threading.RLock()
snapshot_count = sum(1 for _p in SNAP_DIR.iterdir()
                     if _p.is_file() and _p.suffix.lower() in (".jpg", ".jpeg", ".png"))

# ====== 录制（录屏回放） ======
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
    """供手机端回放列表使用：返回录像与近期照片。"""
    items = []
    try:
        for f in sorted(REC_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                items.append({"name": f.name, "type": "video",
                              "url": f"/file/{f.name}", "size": f.stat().st_size})
        for f in sorted(SNAP_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append({"name": f.name, "type": "image",
                              "url": f"/snap/{f.name}", "size": f.stat().st_size})
    except Exception:
        pass
    return items

pending_main_actions = collections.deque()


def phone_command(action):
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
    return "unknown"

def get_max_storage_bytes():
    return MAX_STORAGE_GB * 1024 * 1024 * 1024

def get_dir_size_mb():
    """Get total size of snapshot directory in MB."""
    total = 0
    if SNAP_DIR.exists():
        for f in SNAP_DIR.iterdir():
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 * 1024)

def cleanup_old_snapshots():
    """Delete oldest files until under max storage limit.
    FIFO: oldest files are removed first."""
    max_bytes = get_max_storage_bytes()
    files = sorted(SNAP_DIR.iterdir(), key=lambda f: f.stat().st_mtime)
    while files and get_dir_size_mb() * 1024 * 1024 > max_bytes * 0.95:
        oldest = files.pop(0)
        try:
            oldest.unlink()
            print(f"  [清理] 已删除最旧照片: {oldest.name} ({get_dir_size_mb():.1f} MB)")
        except:
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
    """Save a snapshot with timestamp filename, manage storage.
    full_frame=True: save entire frame with watermark.
    full_frame=False: save cropped detection."""
    global last_snap_path, snapshot_count
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
        with open(str(fpath), "wb") as _f:
            _f.write(buf.tobytes())
        size_mb = get_dir_size_mb()
        print(f"  [拍照] {fname}  ({size_mb:.1f} MB / {MAX_STORAGE_GB} GB)")
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


def maybe_auto_snapshot(source_frame, boxes, camera_key, source_label=None):
    """Capture the best person, with movement and periodic fallback triggers."""
    global last_snap_target, last_snap_time, snap_flash
    if (not detection_enabled or not auto_snap or paused or sel or not focus or
            focus_id < 0 or source_frame is None):
        return False

    now_time = time.time()
    state = auto_snap_states.setdefault(camera_key, {
        'last_time': 0.0,
        'last_target': None,
        'missing_since': None,
        'absence_reset': False,
    })
    frame_h, frame_w = source_frame.shape[:2]
    xyxy_list, conf_list, cls_list, track_id_list = get_boxes_data(boxes)

    best = None
    best_score = -1.0
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
        area_score = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
        score = conf + (0.15 if track_id >= 0 else 0.0) + area_score
        if score > best_score:
            best_score = score
            best = (cid, conf, x1, y1, x2, y2)

    if best is None:
        if state['missing_since'] is None:
            state['missing_since'] = now_time
        elif (not state['absence_reset'] and
              now_time - state['missing_since'] >= snap_absence_reset):
            # A person leaving and returning is a new event, even if the previous
            # photo was less than the normal stationary-person interval ago.
            state['last_target'] = None
            state['absence_reset'] = True
        return False

    state['missing_since'] = None
    state['absence_reset'] = False
    cid, conf, x1, y1, x2, y2 = best
    curr_target = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    elapsed = now_time - state['last_time']
    should_snap = (state['last_target'] is None and
                   (state['last_time'] <= 0.0 or elapsed >= snap_cooldown))

    if not should_snap and elapsed >= snap_max_interval:
        # Stationary people still need periodic evidence; the previous logic
        # could suppress all photos forever after the first one.
        should_snap = True
    elif not should_snap and elapsed >= snap_cooldown:
        prev_cx, prev_cy, prev_w, prev_h = state['last_target']
        curr_cx, curr_cy, curr_w, curr_h = curr_target
        move_dist = (((curr_cx - prev_cx) / max(frame_w, 1)) ** 2 +
                     ((curr_cy - prev_cy) / max(frame_h, 1)) ** 2) ** 0.5
        prev_area = prev_w * prev_h
        curr_area = curr_w * curr_h
        size_change = abs(curr_area - prev_area) / max(prev_area, 1)
        should_snap = (move_dist > snap_move_threshold or
                       size_change > snap_size_threshold)

    if not should_snap:
        return False

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
    state['last_target'] = curr_target
    last_snap_time = now_time
    last_snap_target = curr_target
    snap_flash = 12
    print(f"  [自动拍] {source_label or '当前摄像头'} 已保存")
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
            if not verify or _read_valid_camera_frame(cap) is not None:
                return cap, backend
        cap.release()
    return cv2.VideoCapture(), None


def get_camera_devices(active_index=None):
    """Enumerate cameras that can continuously deliver valid frames."""
    devices = []
    if active_index is not None:
        devices.append({
            'index': int(active_index),
            'label': f"摄像头{active_index} (当前)",
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
                'label': f"摄像头{i} ({backend_name})",
            })
        test_cap.release()
    devices.sort(key=lambda item: item['index'])
    return devices

def get_cam_names(active_index=None):
    devices = get_camera_devices(active_index=active_index)
    return [device['label'] for device in devices] if devices else [f"摄像头{i}" for i in range(3)]


def open_cam(idx):
    cap, backend = try_open_camera(idx)
    target_w, target_h = 1280, 720
    current_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0
    current_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0
    # DirectShow rebuilds its graph on FOURCC/height/FPS changes (several seconds
    # per camera). Keep an already-HD native mode; only negotiate when resolution
    # is genuinely below the requested monitoring quality.
    if cap.isOpened() and (current_w < target_w or current_h < target_h):
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
    """屏幕上的可点击按钮，带平滑悬停高亮与点击脉冲反馈。"""
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
    """管理所有可点击按钮，并维护点击脉冲动画状态。"""
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
            btn.hover_amt += (target - btn.hover_amt) * 0.28  # 指数平滑
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


# ====== Multi-Camera (多摄平铺) ======
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
    """每路摄像头独立采集；停止时绝不在 ``read`` 中途释放句柄。"""
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
            print(f"多摄: 摄像头 {idx} 无有效画面，已跳过")
            continue
        opened.append(CamCaptureThread(camera_cap, idx, dev['label']))

    total = len(opened) + (1 if reuse_capture is not None and reuse_capture.isOpened() else 0)
    if total < 2:
        for th in opened:
            th.request_stop()
            th.release_after_join()
        print(f"多摄: 只成功打开 {total} 路，需要至少 2 路")
        return False

    if reuse_capture is not None and reuse_capture.isOpened():
        label = next((d['label'] for d in cam_devices if d['index'] == reuse_index),
                     f"摄像头{reuse_index}")
        opened.insert(0, CamCaptureThread(reuse_capture, reuse_index, label))

    multi_threads = opened
    multi_cached_results = [
        {'boxes': np.empty((0, 4)), 'conf': np.array([]), 'cls': np.array([])}
        for _ in opened
    ]
    for th in multi_threads:
        th.start()
    print(f"多摄模式: 已打开 {len(multi_threads)} 个摄像头（独立采集线程）")
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


def do_toggle_multi_cam():
    """稳定切换多摄/单摄；过滤连点并串行化所有摄像头资源变更。"""
    global multi_cam, _buttons_built_for_size, cap, cur, cam_w, cam_h, cam_fps
    global multi_frame_counter, camera_switching, last_multi_toggle_at

    now = time.monotonic()
    if now - last_multi_toggle_at < MULTI_TOGGLE_DEBOUNCE:
        print("多摄: 操作过快，已忽略重复点击")
        return False
    if camera_scan_running:
        print("多摄: 正在扫描摄像头，请稍候再试")
        return False
    if not camera_transition_lock.acquire(blocking=False):
        print("多摄: 上一次切换尚未完成")
        return False

    camera_switching = True
    _buttons_built_for_size = None
    try:
        if not multi_cam:
            if len(cam_devices) < 2:
                print("多摄: 可用摄像头不足 2 个，请先刷新列表")
                return False
            start_cam_transition()
            if not open_all_cameras(reuse_capture=cap, reuse_index=cur):
                print("多摄: 打开失败，保持单摄")
                return False
            multi_cam = True
            multi_frame_counter = 0
            reset_auto_snap_state()
            print(f"多摄模式: {len(multi_threads)} 个摄像头平铺")
        else:
            start_cam_transition()
            kept_cap = close_all_cameras(keep_index=cur)
            if kept_cap is False:
                print("多摄: 线程尚未安全退出，保持多摄模式")
                return False
            if kept_cap is None or not kept_cap.isOpened():
                print("多摄: 句柄交接失败，尝试重新打开单摄")
                kept_cap, cam_w, cam_h, cam_fps = open_cam(cur)
                if not kept_cap.isOpened():
                    kept_cap.release()
                    print("多摄: 单摄恢复失败，正在重新打开多摄")
                    if open_all_cameras():
                        multi_cam = True
                    return False
            cap = kept_cap
            cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or OUTPUT_W
            cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or OUTPUT_H
            cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            multi_cam = False
            reset_auto_snap_state()
            print("已切回单摄模式")
        return True
    except Exception as exc:
        print(f"多摄切换失败: {exc}")
        if not multi_cam and (cap is None or not cap.isOpened()):
            cap, cam_w, cam_h, cam_fps = open_cam(cur)
        return False
    finally:
        last_multi_toggle_at = time.monotonic()
        camera_switching = False
        _buttons_built_for_size = None
        camera_transition_lock.release()


# ====== 性能优化配置 ======
# 推理输入尺寸：GPU 算力足用较大尺寸提升小目标识别率；CPU 适当降低保速度
INFER_SIZE = 640
# 跳帧推理：GPU 逐帧推理(=1)以保证实时性与识别率；CPU 每 2 帧推理一次
INFER_EVERY_N = 1 if device == "cuda" else 2
# 多摄每路推理尺寸（每帧要跑多路，故偏小保证整体流畅）
MULTI_INFER_SIZE = 416 if device == "cuda" else 320
# 使用半精度推理（需 GPU 支持）
USE_HALF = device == "cuda"
OUTPUT_W, OUTPUT_H = 1280, 720
PHONE_STACK_W = 960      # 手机端多摄竖排时每路画面的宽度（竖排避免平铺过小）
PHONE_INFO_PANEL = _os.environ.get("YOLO_PHONE_PANEL", "0").lower() in {"1", "true", "yes", "on"}
STREAM_FPS = int(_os.environ.get("YOLO_STREAM_FPS", "30"))
STREAM_QUALITY = int(_os.environ.get("YOLO_STREAM_QUALITY", "68"))
STREAM_MAX_WIDTH = int(_os.environ.get("YOLO_STREAM_WIDTH", "960"))



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
    """把单摄横屏画面合成为竖屏自适应画面。

    中间放摄像头画面，上下留白区改成信息面板（时间/人数/帧率/锁定状态/
    在线设备/最近抓拍缩略图），这样手机竖屏观看时原本的黑边被利用起来，
    画面更饱满、信息更直观。
    """
    W, H = 720, 1280
    ch, cw = cam.shape[:2]
    cam_h = max(1, int(round(W * ch / max(cw, 1))))
    cam_r = cv2.resize(cam, (W, cam_h), interpolation=cv2.INTER_LINEAR)
    H_top = (H - cam_h) // 2
    H_bot = H - H_top - cam_h
    # Fill the portrait screen with a blurred live background instead of black bars,
    # while keeping the complete uncropped camera image sharp in the center.
    tiny = cv2.resize(cam, (60, 106), interpolation=cv2.INTER_AREA)
    tiny = cv2.GaussianBlur(tiny, (0, 0), 3.5)
    canvas = cv2.resize(tiny, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.addWeighted(canvas, 0.34, np.zeros_like(canvas), 0.66, 0)
    canvas[H_top:H_top + cam_h, 0:W] = cam_r
    # 上下分隔亮条（青绿）
    cv2.rectangle(canvas, (0, H_top - 4), (W, H_top), (120, 180, 0), -1)
    cv2.rectangle(canvas, (0, H_top + cam_h), (W, H_top + cam_h + 4), (120, 180, 0), -1)

    # ---- 顶部信息面板 ----
    now = datetime.datetime.now()
    canvas = draw_cn(canvas, "实时监控", (24, 28), 34, (0, 230, 150))
    canvas = draw_cn(canvas, "人体识别 · 手机远程", (24, 78), 18, (150, 160, 170))
    canvas = draw_cn(canvas, now.strftime("%Y-%m-%d"), (W - 24, 26), 20, (200, 200, 205), anchor="rt")
    canvas = draw_cn(canvas, now.strftime("%H:%M:%S"), (W - 24, 56), 30, (255, 255, 255), anchor="rt")

    # ---- 底部信息面板 ----
    by = H_top + cam_h + 16
    # 左：最近抓拍缩略图
    canvas = draw_cn(canvas, "最近抓拍", (24, by + 4), 18, (150, 200, 255))
    if snap_thumb is not None:
        canvas[by + 28:by + 28 + 94, 24:24 + 140] = snap_thumb
        cv2.rectangle(canvas, (24, by + 28), (24 + 140, by + 28 + 94), (90, 110, 130), 1)
    else:
        canvas = draw_cn(canvas, "暂无抓拍", (24, by + 60), 18, (110, 120, 130))
    # 中：检测人数 + 监控状态
    canvas = draw_cn(canvas, "当前检测人数", (190, by + 6), 18, (170, 180, 190))
    cnt_color = (0, 230, 150) if person_count > 0 else (120, 130, 140)
    canvas = draw_cn(canvas, f"{person_count}", (190, by + 30), 54, cnt_color)
    dot = (0, 220, 120) if person_count > 0 else (100, 110, 120)
    cv2.circle(canvas, (196, by + 108), 8, dot, -1)
    canvas = draw_cn(canvas, "监控中", (210, by + 100), 18, (180, 190, 200))
    # 右：在线设备 + 帧率 + 锁定状态
    canvas = draw_cn(canvas, f"在线设备 {online}", (W - 24, by + 6), 20, (120, 200, 255), anchor="rt")
    canvas = draw_cn(canvas, f"帧率 {fps:.1f} FPS", (W - 24, by + 44), 20, (255, 230, 120), anchor="rt")
    if locked_id is not None:
        canvas = draw_cn(canvas, f"已锁定 #{locked_id}", (W - 24, by + 82), 20, (0, 230, 150), anchor="rt")
        canvas = draw_cn(canvas, f"置信度 {locked_conf:.2f}", (W - 24, by + 112), 18, (200, 210, 220), anchor="rt")
    else:
        canvas = draw_cn(canvas, "未锁定目标", (W - 24, by + 82), 20, (150, 160, 170), anchor="rt")
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
cam_devices = [{'index': cur, 'label': f"摄像头{cur}"}]
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
inp = ""
sel = False
sel_page = 0
paused = False
show_help = False
auto_snap = False  # 默认关闭自动拍照，避免开机即抓拍
snap_cooldown = float(_os.environ.get("YOLO_SNAP_COOLDOWN", "10"))
snap_max_interval = float(_os.environ.get("YOLO_SNAP_MAX_INTERVAL", "30"))
snap_absence_reset = float(_os.environ.get("YOLO_SNAP_ABSENCE_RESET", "3"))
last_snap_time = 0.0
snap_flash = 0
snap_mode = "crop"   # "crop" = 拍检测物体, "full" = 全屏拍照(带水印)
last_snap_path = None  # 最近一次抓拍的绝对路径（用于手机端底部缩略图）

# ====== 手机同步观看（局域网 MJPEG 推流）======
STREAM_PORT = 8000
streamer = MJPEGStreamer(port=STREAM_PORT, quality=STREAM_QUALITY, fps_cap=STREAM_FPS,
                         max_width=STREAM_MAX_WIDTH, max_height=720,
                         on_command=phone_command,
                         file_root=REC_DIR, snap_root=SNAP_DIR,
                         get_file_list=list_playback_files,
                         get_status=lambda: {"recording": recording,
                                             "clients": streamer.client_count(),
                                             "remote": remote_url,
                                             "lan": streamer.url,
                                             "detection_enabled": detection_enabled,
                                             "auto_snap": auto_snap,
                                             "snapshot_count": snapshot_count,
                                             "last_snapshot": Path(last_snap_path).name if last_snap_path else None,
                                             "snapshot_dir": str(SNAP_DIR),
                                             "camera_switching": camera_switching,
                                             "camera_count": len(cam_devices),
                                             "camera_indices": [d['index'] for d in cam_devices],
                                             "multi_camera": multi_cam,
                                             "active_camera_count": len(multi_threads) if multi_cam else 1,
                                             "qr_target": qr_target,
                                             "qr_url": (remote_url if qr_target == "public" and remote_url else streamer.url),
                                             "stream_fps": streamer.encoded_fps,
                                             "encode_ms": streamer.encode_ms,
                                             "jpeg_kb": streamer.jpeg_kb,
                                             "stream_width": STREAM_MAX_WIDTH,
                                             "stream_quality": STREAM_QUALITY})
streamer.start()
show_stream_qr = True   # 是否在窗口上叠加二维码/地址
qr_target = "public"    # 二维码默认指向公网（隧道连上后自动显示公网二维码）
remote_url = None       # 公网访问地址（由隧道回调填入）
_qr_img = make_qr_image(streamer.url)  # 连上前先用局域网二维码；连上后自动切公网
stream_url = streamer.url


def set_remote_url(u):
    """隧道建立后回调：记录公网地址，并在二维码指向公网时刷新。"""
    global remote_url, _qr_img, qr_target
    remote_url = u
    print("\n[公网] 远程观看地址: " + u)
    print("（手机用流量/异地网络打开此链接即可，无需同一 WiFi）\n")
    if qr_target == "public":
        _qr_img = make_qr_image(u)
    # The tunnel callback runs on a worker thread. Never call OpenCV/Win32
    # window APIs here; doing so can deadlock the main thread inside imshow().


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
    _qr_img = make_qr_image(remote_url if qr_target == "public" else streamer.url)
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

# Model warmup is deferred until the first inference, after the window is visible.

# 智能拍照：记录上次拍照的目标状态（位置+大小），用于判断是否需要补拍
last_snap_target = None  # (cx, cy, w, h) 最近一次自动抓拍目标
snap_move_threshold = 0.25  # 目标移动超过画面25%或尺寸变化超过50%时补拍
snap_size_threshold = 0.5
auto_snap_states = {}       # 每个摄像头独立维护冷却、离场与周期补拍状态

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

# ====== 多摄像头平铺模式 ======
multi_cam = False          # 是否处于多摄平铺模式
multi_threads = []         # [CamCaptureThread, ...] 每路独立采集线程
multi_cached_results = []  # 每路摄像头的推理缓存 [{boxes, conf, cls}, ...]
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

    threading.Thread(target=worker, name="camera-scan", daemon=True).start()

def do_toggle_detection():
    global detection_enabled, cached_boxes, g_boxes
    global locked_track_id, locked_box, locked_conf, locked_misses
    global auto_snap, _buttons_built_for_size, multi_cached_results
    detection_enabled = not detection_enabled
    cached_boxes = {
        'xyxy': np.empty((0, 4)), 'conf': np.array([]),
        'cls': np.array([]), 'track_id': np.array([]),
    }
    g_boxes = cached_boxes
    locked_track_id = None
    locked_box = None
    locked_conf = 0.0
    locked_misses = 0
    multi_cached_results = [
        {'boxes': np.empty((0, 4)), 'conf': np.array([]), 'cls': np.array([])}
        for _ in multi_cached_results
    ]
    if not detection_enabled:
        auto_snap = False
    else:
        start_model_warm_async()
    reset_auto_snap_state()
    _buttons_built_for_size = None
    print("Detection mode: ON" if detection_enabled else "Monitor mode: ON (inference disabled)")


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
        print("  [拍照] 尚未检测到画面")
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
            save_snapshot(g_frame, "无目标", 0.0)
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
        icon.notify(msg, "YOLO 人体识别")
    except Exception:
        pass


def setup_tray():
    global tray_icon
    try:
        import pystray
        from pystray import Menu, MenuItem
        menu = Menu(
            MenuItem("显示画面", _tray_show),
            MenuItem("隐藏画面", _tray_hide),
            MenuItem("复制观看地址", _tray_copy_url),
            Menu.SEPARATOR,
            MenuItem("退出", _tray_exit),
        )
        tray_icon = pystray.Icon("yolo_cam_tray", _make_tray_image(),
                                 "YOLO 人体识别", menu)
        import threading as _th
        _th.Thread(target=tray_icon.run, daemon=True).start()
        print("[托盘] 已启动系统托盘图标（后台隐藏运行），右键图标可显示画面或退出")
    except Exception as e:
        global show_desktop
        show_desktop = True   # 托盘不可用则退回可见窗口，避免程序隐形且无法退出
        print(f"[托盘] 启动失败，退回普通窗口模式（按 Q 或点窗口关闭退出）：{e}")


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
                 multi_cam, camera_switching)
    if _buttons_built_for_size == cache_key and _buttons_cache is not None:
        btn_mgr.buttons = _buttons_cache.copy()
        return

    btn_mgr.clear()
    _buttons_built_for_size = cache_key

    # ====== Normal mode ======
    current_device = next((device for device in cam_devices if device['index'] == cur), None)
    cam_display = current_device['label'] if current_device else f"摄像头{cur}"
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
            "扫描中" if camera_scan_running else "刷新列表", do_refresh_cameras,
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
    fn_x = w - fn_w - 10
    fn_start_y = h - btn_h - 8
    fn_labels = [
        ("切到监控" if detection_enabled else "开启识别", do_toggle_detection, not detection_enabled, (46, 132, 104), False),
        ("拍照", do_manual_snap, True, (56, 132, 92), False),
        ("手机投屏", do_toggle_stream_qr, show_stream_qr, (48, 118, 148), False),
        ("录制" if not recording else "停止录", toggle_recording, recording, (168, 64, 64), False),
        ("多摄", do_toggle_multi_cam, multi_cam, (128, 80, 168), False),
        ("自动拍", do_toggle_auto_snap, auto_snap, (148, 118, 48), False),
        ("全屏拍" if snap_mode == "full" else "人物拍", do_toggle_snap_mode, snap_mode == "full", (88, 96, 148), False),
        ("窗口化" if full else "全屏", toggle_full, full, (76, 76, 76), False),
        ("继续" if paused else "暂停", do_toggle_pause, paused, (76, 76, 76), False),
        ("帮助", do_toggle_help, show_help, (76, 76, 76), False),
        ("退出", do_exit, False, (150, 58, 58), True),
    ]
    for i, (label, cb, is_active, color, is_danger) in enumerate(fn_labels):
        base_color = (156, 62, 62) if is_danger else ((52, 138, 90) if is_active else (72, 72, 72))
        btn_mgr.add(Button(
            fn_x, fn_start_y - (i + 1) * (btn_h + gap), fn_w, btn_h,
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
print("  M:Snapshot mode  V:Toggle phone panel  D:Detection/Monitor")
print("====================\n")

# 实时帧率追踪
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
    phone_frame_offered = False  # reset once per display loop
    # ====== 多摄像头平铺模式（跳过单摄读取） ======
    phone_canvas = None   # 手机端专用画面（默认 None，下面分支会赋值）
    if multi_cam:
        phone_canvas = None   # 手机端专用画面（干净、无桌面 UI），多摄时上下竖排
        phone_stack = []      # 多摄各路的干净画面，竖排拼接用
        multi_frame_counter += 1
        multi_do_infer = detection_enabled and model_warmed and (multi_frame_counter % MULTI_TILE_INFER_N == 0)
        raw_frames = read_all_frames()
        rows, cols = calc_grid(len(multi_threads))
        tile_w = OUTPUT_W // cols
        tile_h = OUTPUT_H // rows
        canvas = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)

        for ti, th in enumerate(multi_threads):
            r = ti // cols
            c = ti % cols
            tx = c * tile_w
            ty = r * tile_h
            tile_frame = raw_frames[ti]
            label = th.label
            # 保持宽高比缩放 + 黑边填充（letterbox）
            fh, fw = tile_frame.shape[:2]
            tile_area, scale, ox, oy, new_w, new_h = fit_complete_on_blur(
                tile_frame, tile_w, tile_h)
            canvas[ty:ty+tile_h, tx:tx+tile_w] = tile_area

            if multi_do_infer:
                try:
                    # 用 predict 直接返回原图坐标系下的框，避免跨摄像头 tracker 串号
                    tresults = model.predict(
                        tile_frame,
                        imgsz=MULTI_INFER_SIZE,
                        verbose=False,
                        half=USE_HALF,
                        classes=[focus_id],
                        conf=PERSON_CONF,
                        iou=PERSON_IOU,
                    )
                    tboxes = tresults[0].boxes
                    if len(tboxes) > 0:
                        txyxy = tboxes.xyxy.clone().cpu().numpy()
                        tconf = tboxes.conf.clone().cpu().numpy()
                        tcls = tboxes.cls.clone().cpu().numpy()
                    else:
                        txyxy, tconf, tcls = np.empty((0, 4)), np.array([]), np.array([])
                    multi_cached_results[ti] = {'boxes': txyxy, 'conf': tconf, 'cls': tcls}
                except Exception:
                    multi_cached_results[ti] = {'boxes': np.empty((0, 4)),
                                               'conf': np.array([]), 'cls': np.array([])}

            # 绘制检测框（框在原始 tile 坐标，按显示缩放+偏移映射到画布）
            res = multi_cached_results[ti]
            # 每路摄像头使用自己的原始画面、检测结果和冷却状态自动抓拍。
            maybe_auto_snapshot(tile_frame, res, ("camera", th.idx), f"摄像头{th.idx}")
            if 'boxes' in res and res['boxes'] is not None and len(res['boxes']) > 0:
                sx = new_w / max(fw, 1)
                sy = new_h / max(fh, 1)
                for bi in range(len(res['boxes'])):
                    x1, y1, x2, y2 = map(int, res['boxes'][bi])
                    conf = float(res['conf'][bi]) if len(res['conf']) > bi else 0
                    x1 = int(x1 * sx) + tx + ox
                    y1 = int(y1 * sy) + ty + oy
                    x2 = int(x2 * sx) + tx + ox
                    y2 = int(y2 * sy) + ty + oy
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 120), 2)
                    canvas = draw_cn(canvas, f"人 {conf:.2f}", (x1 + 4, y1 - 6), 12, (0, 0, 0), (0, 220, 120))

            # 手机端专用干净画面：仅摄像头+检测框，不带桌面 UI（按钮/标题/信息面板）。
            # 多摄时改为上下竖排，避免平铺导致每路画面过小、看不清。
            _pt = tile_frame.copy()
            if 'boxes' in res and res['boxes'] is not None and len(res['boxes']) > 0:
                for bi in range(len(res['boxes'])):
                    _bx1, _by1, _bx2, _by2 = map(int, res['boxes'][bi])
                    cv2.rectangle(_pt, (_bx1, _by1), (_bx2, _by2), (0, 220, 120), 2)
                    _conf = float(res['conf'][bi]) if len(res['conf']) > bi else 0
                    _pt = draw_cn(_pt, f"人 {_conf:.2f}", (_bx1 + 4, _by1 - 6), 12, (0, 0, 0), (0, 220, 120))
            _ps = PHONE_STACK_W / max(fw, 1)
            _pt = cv2.resize(_pt, (PHONE_STACK_W, int(fh * _ps)), interpolation=cv2.INTER_LINEAR)
            phone_stack.append(_pt)

            # 摄像头信息：标签 + 时间 + 分辨率 + 各路独立帧率
            cv2.rectangle(canvas, (tx, ty), (tx + 196, ty + 44), (0, 0, 0), -1)
            canvas = draw_cn(canvas, label, (tx + 4, ty + 2), 13, (0, 255, 0))
            multi_now = datetime.datetime.now()
            time_str = multi_now.strftime("%H:%M:%S")
            canvas = draw_cn(canvas, time_str, (tx + tile_w - 6, ty + 2), 12, (255, 255, 200), anchor="rt")
            cap_w_ = int(th.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or fw
            cap_h_ = int(th.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fh
            res_str = f"{cap_w_}x{cap_h_}"
            fps_val = th.fps
            info_str = f"{res_str}  {fps_val:.1f}fps"
            canvas = draw_cn(canvas, info_str, (tx + 4, ty + 22), 11, (200, 200, 200))
            if th.error:
                canvas = draw_cn(canvas, "无信号", (tx + 4, ty + 38), 11, (255, 120, 120))

        # 手机端多摄画面（干净、无桌面 UI）
        if streamer.phone_layout == "tile":
            # 横屏/全屏：平铺，每路占满宽，黑边更少
            _rows, _cols = calc_grid(len(multi_threads))
            _tw, _th = 1280 // _cols, 720 // _rows
            pt_canvas = np.zeros((720, 1280, 3), np.uint8)
            for ti, th_ in enumerate(multi_threads):
                r = ti // _cols
                c = ti % _cols
                _tx = c * _tw
                _ty = r * _th
                _f = raw_frames[ti]
                fh, fw = _f.shape[:2]
                _tile, _s, _ox, _oy, _nw, _nh = fit_complete_on_blur(_f, _tw, _th)
                pt_canvas[_ty:_ty + _th, _tx:_tx + _tw] = _tile
                _res = multi_cached_results[ti]
                if 'boxes' in _res and _res['boxes'] is not None and len(_res['boxes']) > 0:
                    _sx, _sy = _nw / max(fw, 1), _nh / max(fh, 1)
                    for bi in range(len(_res['boxes'])):
                        x1, y1, x2, y2 = map(int, _res['boxes'][bi])
                        x1 = int(x1 * _sx) + _tx + _ox
                        y1 = int(y1 * _sy) + _ty + _oy
                        x2 = int(x2 * _sx) + _tx + _ox
                        y2 = int(y2 * _sy) + _ty + _oy
                        cv2.rectangle(pt_canvas, (x1, y1), (x2, y2), (0, 220, 120), 2)
            phone_canvas = pt_canvas
        else:
            # 竖屏：上下竖排拼接（无桌面 UI）
            if phone_stack:
                phone_canvas = np.vstack(phone_stack)

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

        # 全局 FPS（用于日期下方显示）
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_realtime = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
    elif not paused:
        ret, frame = cap.read()
        if not ret:
            start_cam_transition()   # 设备断流重连：旧画面缓出，新画面淡入
            cap.release()
            cap, cam_w, cam_h, cam_fps = open_cam(cur)
            ret, frame = cap.read()
        if not ret:
            break

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
        # 跳帧推理：只在指定间隔推理，中间帧复用缓存
        do_infer = detection_enabled and model_warmed and (frame_counter % INFER_EVERY_N == 0)

        if do_infer:
            if not _first_inference_logged:
                print("First live inference started.")
            _infer_t0 = time.perf_counter()
            # 直接传入整帧并指定 imgsz：ultralytics 返回原图坐标系下的框，无需手动缩放
            results = model.track(
                frame,
                imgsz=INFER_SIZE,
                verbose=False,
                half=USE_HALF,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[focus_id],
                conf=PERSON_CONF,
                iou=PERSON_IOU,
            )
            if not _first_inference_logged:
                print(f"First live inference finished in {(time.perf_counter() - _infer_t0) * 1000:.1f} ms.")
                _first_inference_logged = True
            boxes = results[0].boxes
            if len(boxes) > 0:
                track_id = []
                if getattr(boxes, 'id', None) is not None:
                    track_id = boxes.id.int().cpu().numpy()
                cached_boxes = {
                    'xyxy': boxes.xyxy.clone().cpu().numpy(),
                    'conf': boxes.conf.clone().cpu().numpy() if len(boxes.conf) > 0 else np.array([]),
                    'cls': boxes.cls.clone().cpu().numpy() if len(boxes.cls) > 0 else np.array([]),
                    'track_id': track_id,
                }
            else:
                cached_boxes = {
                    'xyxy': np.empty((0, 4)),
                    'conf': np.array([]),
                    'cls': np.array([]),
                    'track_id': np.array([]),
                }
        else:
            boxes = cached_boxes

        g_frame = frame.copy()
        g_boxes = boxes
        ann = frame.copy()
        # 摄像头切换：新画面交叉淡入（旧画面缓出），消除硬切卡顿感
        if cam_transition['active'] and cam_transition['old'] is not None:
            _p = (time.time() - cam_transition['t0']) / cam_transition['dur']
            if _p >= 1.0:
                cam_transition['active'] = False
            else:
                _e = _p * _p * (3 - 2 * _p)   # smoothstep 缓动
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

        # 手机端专用画面：竖屏自适应——摄像头居中，上下留白区显示识别统计。
        # 仅在有手机观看时才合成，避免无人观看时白白消耗 CPU。
        if not phone_frame_offered:
            if streamer.client_count() > 0:
                if streamer.phone_layout == "stack" or (detection_enabled and PHONE_INFO_PANEL):
                    _snap_thumb = get_snap_thumb()
                    phone_canvas = make_phone_portrait(
                        ann,
                        person_count=len(candidates),
                        fps=fps_realtime,
                        locked_id=locked_track_id,
                        locked_conf=locked_conf,
                        snap_thumb=_snap_thumb,
                        online=streamer.client_count(),
                    )
                else:
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
        if paused:
            ann = draw_cn(ann, "[已暂停] 按P继续", (w // 2 - 100, 30), 24,
                          (0, 255, 255), (0, 0, 0))

        if not detection_enabled:
            title_txt = "[纯监控模式] 已关闭模型识别, 按D 恢复识别"
        elif not model_warmed:
            title_txt = "[模型加载中] 摄像头画面已正常显示"
        else:
            title_txt = "[人体识别] M:全屏拍照/人物抓拍"
        if multi_cam:
            # 多摄：标题移到顶部居中，避免压住第一路摄像头左上角的信息面板
            try:
                _tf = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
                _bb = _tf.getbbox(title_txt)
                _tw = _bb[2] - _bb[0]
            except Exception:
                _tw = len(title_txt) * 14
            ann = draw_cn(ann, title_txt, (max(10, w // 2 - _tw // 2), 10),
                          16, (0, 255, 0), (30, 30, 30))
        else:
            ann = draw_cn(ann, title_txt, (10, 28), 16, (0, 255, 0), (30, 30, 30))

        if recording and not show_stream_qr:
            ann = draw_cn(ann, "● 录制中", (w // 2 - 44, 30), 18, (255, 90, 90), (40, 0, 0))

        # ====== Build and draw buttons ======
        btn_mgr.tick_animations(time.time())
        build_buttons(w, h)
        ann = btn_mgr.draw_all(ann, time.time())

        # Help overlay
        if show_help and not sel:
            help_texts = [
                "===== 帮助 =====",
                "Q:退出 F:全屏 P:暂停",
                "左侧两列卡片可直接选择工作摄像头",
                "点“刷新列表”可重新扫描插拔后的摄像头",
                "C/X:备用切换 1:锁定人物 H:帮助",
                "Space:拍照 R:录制/停止 B:二维码切换(局域网/公网)",
                "M:切换全屏拍照/人物抓拍",
                "手机:可拍照/录制/回放(免同一WiFi也能远程看)",
                "================",
            ]
            overlay_h = len(help_texts) * 26 + 20
            overlay2 = np.full((overlay_h, w, 3), (0, 0, 0), dtype=np.uint8)
            ann[:overlay_h, :] = cv2.addWeighted(ann[:overlay_h, :], 0.4, overlay2, 0.6, 0)
            for i, txt in enumerate(help_texts):
                ann = draw_cn(ann, txt, (20, 10 + i * 26), 18, (0, 255, 255))

        # ====== Date/time display (top-right, won't block buttons) ======
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
        maybe_auto_snapshot(frame, boxes, ("camera", cur), f"摄像头{cur}")

    # Flash effect
    if snap_flash > 0:
        ann = cv2.addWeighted(ann, 1.0, np.full_like(ann, 255), 0.4, 0)
        snap_flash -= 1

    # Auto-snap indicator
    if auto_snap:
        mode_txt = f"[自动拍:{'开' if auto_snap else '关'}|{'全屏' if snap_mode=='full' else '人物'}]"
        ann = draw_cn(ann, mode_txt, (10, 50), 15, (0, 200, 255), (0, 0, 0))


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

    # ====== 手机投屏：窗口上叠加二维码与访问地址 ======
    if show_stream_qr and show_desktop:
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
        # 文本
        tx = panel_x + pad + qr_size + 12
        ann = draw_cn(ann, "手机同步观看", (tx, panel_y + pad + 2), 16, (0, 255, 180))
        ann = draw_cn(ann, f"局域网: {stream_url}", (tx, panel_y + pad + 30), 13, (255, 230, 120))
        line_pub = f"公网: {remote_url}" if remote_url else "公网: 未连接(按B重试)"
        ann = draw_cn(ann, line_pub, (tx, panel_y + pad + 54), 13,
                      (255, 180, 120) if remote_url else (200, 200, 200))
        target_txt = f"在线: {streamer._clients}  [{'公网' if qr_target == 'public' else '局域网'}二维码]"
        ann = draw_cn(ann, target_txt, (tx, panel_y + pad + 80), 12, (150, 220, 255))
        if recording:
            ann = draw_cn(ann, "● 录制中", (tx, panel_y + pad + 104), 13, (255, 90, 90))

    if show_desktop:
        if not window_created:
            _create_desktop_window(full)
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
