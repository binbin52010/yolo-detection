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

BASE_DIR = Path(__file__).parent.resolve()
model_path = BASE_DIR / "models" / "yolov8n.pt"

print(f"Loading model: {model_path}")
model = YOLO(str(model_path))
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {device}")
print(f"OS: {platform.system()} {platform.release()}")

# ====== Font ======
font_path = None
font_candidates = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]
# Also try system font discovery
if platform.system() == "Windows":
    font_dir = Path("C:/Windows/Fonts")
    if font_dir.exists():
        for fp in font_dir.glob("*.ttf"):
            name_lower = fp.name.lower()
            if "msyh" in name_lower or "simhei" in name_lower or "simsun" in name_lower:
                font_candidates.insert(0, str(fp))
                break

for fp in font_candidates:
    if Path(fp).exists():
        font_path = fp
        print(f"Font: {fp}")
        break

if not font_path:
    print("Warning: No Chinese font found, using default")

def draw_cn(img, text, pos, size=18, color=(0, 255, 0), bg=None):
    """Draw Chinese text on an OpenCV BGR image using PIL."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    x, y = pos
    if bg:
        bbox = d.textbbox((0, 0), text, font=font)
        d.rectangle(
            [x, y, x + bbox[2] - bbox[0] + 6, y + bbox[3] - bbox[1] + 4],
            fill=bg
        )
    d.text((x + 2, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

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
cn_list = sorted(COCO_CN.values())
all_names = model.names

# ====== Snapshot (拍照) Settings ======
SNAP_DIR = BASE_DIR / "snapshots"
SNAP_DIR.mkdir(exist_ok=True)
MAX_STORAGE_GB = 10  # Max storage size in GB (adjustable)

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
    """Add timestamp watermark to bottom-right corner of image."""
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    h, w = img.shape[:2]
    # Draw semi-transparent background strip
    overlay = img.copy()
    strip_h = 32
    cv2.rectangle(overlay, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    # Draw text using PIL for Chinese support
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    d.text((w - 280, h - 26), ts, font=font, fill=(255, 255, 200))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def save_snapshot(frame, label, conf, full_frame=False):
    """Save a snapshot with timestamp filename, manage storage.
    full_frame=True: save entire frame with watermark.
    full_frame=False: save cropped detection."""
    cleanup_old_snapshots()

    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d-%H-%M-%S")
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
        # Cropped object
        watermarked = frame
        fname = f"{ts}_{safe_label}_{conf:.2f}.jpg"

    fpath = SNAP_DIR / fname

    try:
        cv2.imwrite(str(fpath), watermarked, [cv2.IMWRITE_JPEG_QUALITY, 85])
        size_mb = get_dir_size_mb()
        print(f"  [拍照] {fname}  ({size_mb:.1f} MB / {MAX_STORAGE_GB} GB)")
        return True
    except Exception as e:
        print(f"  [拍照失败] {e}")
        return False

def pick_best_detection(frame, boxes, valid_cids):
    """
    Pick the best detection for snapshot:
    Priority: highest confidence, then largest area.
    Returns (cropped_frame, cn_label, conf) or None.
    """
    best = None
    best_score = -1
    h, w = frame.shape[:2]

    for i in range(len(boxes)):
        cid = int(boxes.cls[i].item())
        if cid not in valid_cids:
            continue
        conf = boxes.conf[i].item()
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
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
def get_cap_backend():
    """Return the best camera backend for the current OS."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    elif system == "Linux":
        return cv2.CAP_V4L2
    return 0  # default

def get_cam_names():
    """枚举系统中存在的摄像头及其名称。"""
    names = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 尝试读取一帧以确认摄像头可用
            ret, _ = cap.read()
            if ret:
                # 尝试获取摄像头名称（部分驱动支持）
                name = cap.getBackendName()
                names.append(f"摄像头{i} ({name})")
            cap.release()
        else:
            cap.release()
    return names if names else [f"摄像头{i}" for i in range(3)]

def open_cam(idx):
    backend = get_cap_backend()
    cap = cv2.VideoCapture(idx, backend)
    # 获取摄像头支持的最大分辨率
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 尝试设置为最大分辨率
    for w, h in [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            cap_w, cap_h = w, h
            break
    cap.set(cv2.CAP_PROP_FPS, 30)
    if cap.isOpened():
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"Camera {idx}: {cap_w}x{cap_h} @ {cap_fps}fps")
    return cap, cap_w, cap_h, cap_fps

def switch_cam(d, cam_names=None):
    global cur, cap, cam_w, cam_h, cam_fps
    cap.release()
    cur = (cur + d) % 10
    new, nw, nh, nfps = open_cam(cur)
    if new.isOpened():
        cap = new
        cam_w, cam_h, cam_fps = nw, nh, nfps
    else:
        # If the forward direction failed, try backward
        cur = (cur - d) % 10
        new, nw, nh, nfps = open_cam(cur)
        if new.isOpened():
            cap = new
            cam_w, cam_h, cam_fps = nw, nh, nfps

# ====== Mouse Clickable Buttons ======
class Button:
    """A clickable button region on the screen."""
    def __init__(self, x, y, w, h, label, callback, color=(60, 60, 60),
                 text_color=(255, 255, 255), font_size=14):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.callback = callback
        self.color = color
        self.text_color = text_color
        self.font_size = font_size
        self.hover = False

    def contains(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, img):
        # Draw button background
        bg = list(self.color)
        if self.hover:
            bg = [min(255, c + 40) for c in bg]
        cv2.rectangle(img, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), (120, 120, 120), 1)
        # Draw text centered
        label = str(self.label)
        img = draw_cn(img, label,
                      (self.x + 6, self.y + (self.h - self.font_size) // 2),
                      size=self.font_size, color=self.text_color)
        return img


class ButtonManager:
    """Manage all clickable buttons."""
    def __init__(self):
        self.buttons = []

    def add(self, btn):
        self.buttons.append(btn)

    def clear(self):
        self.buttons = []

    def handle_click(self, mx, my):
        for btn in self.buttons:
            if btn.contains(mx, my):
                btn.callback()
                return True
        return False

    def update_hover(self, mx, my):
        for btn in self.buttons:
            btn.hover = btn.contains(mx, my)

    def draw_all(self, img):
        for btn in self.buttons:
            img = btn.draw(img)
        return img


# ====== State ======
cam_names = get_cam_names()
cur = 0
cap, cam_w, cam_h, cam_fps = open_cam(cur)
if not cap.isOpened():
    print("Cannot open camera")
    input()
    exit(1)

win = "YOLOv8"
full = False
focus = False
focus_cn = ""
focus_id = -1
inp = ""
sel = False
sel_page = 0
paused = False
show_help = False
auto_snap = True  # 默认开启自动拍照（检测到人物且置信度高时拍一张）
snap_cooldown = 10.0  # 人物检测拍照冷却时间（秒），避免连续拍照
last_snap_time = 0.0
snap_flash = 0
snap_mode = "crop"   # "crop" = 拍检测物体, "full" = 全屏拍照(带水印)

# 智能拍照：记录上次拍照的目标状态（位置+大小），用于判断是否需要补拍
last_snap_target = None  # (cx, cy, w, h) 上次拍照时目标的中心点和尺寸
snap_move_threshold = 0.25  # 目标移动超过画面25%或尺寸变化超过50%时补拍
snap_size_threshold = 0.5

# Latest detection results (updated each loop for snapshot functions)
g_frame = None
g_boxes = None

# Quick items
quick_items = ["人", "猫", "狗", "车", "手机", "椅子", "书", "鸟"]
quick_icons = ["👤", "🐱", "🐶", "🚗", "📱", "🪑", "📖", "🐦"]

btn_mgr = ButtonManager()


def toggle_full():
    global full
    full = not full
    cv2.destroyWindow(win)
    if full:
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)
    cv2.setMouseCallback(win, mouse_callback)


def do_switch_cam(d):
    global cam_names
    switch_cam(d, cam_names)


def do_toggle_focus():
    global focus
    if focus:
        focus = False
        print("Mode: ALL")
    else:
        focus = True
        print("Mode: FOCUS")


def do_toggle_selector():
    global sel, inp, sel_page
    sel = not sel
    inp = ""
    sel_page = 0
    if sel:
        print(f"Selector opened ({len(cn_list)} items)")


def do_toggle_pause():
    global paused
    paused = not paused
    print(f"Paused: {paused}")


def do_toggle_help():
    global show_help
    show_help = not show_help


def do_toggle_auto_snap():
    global auto_snap
    auto_snap = not auto_snap
    print(f"Auto-snap: {'ON' if auto_snap else 'OFF'}")


def do_toggle_snap_mode():
    global snap_mode
    snap_mode = "full" if snap_mode == "crop" else "crop"
    print(f"Snap mode: {'全屏(水印)' if snap_mode == 'full' else '检测物体'}")


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
        # Crop best detection (fallback to full frame if nothing detected)
        valid = {focus_id} if (focus and focus_id >= 0) else set(range(len(all_names)))
        if g_boxes is None or len(g_boxes) == 0:
            # No detections: save full frame without watermark
            save_snapshot(g_frame, "无目标", 0.0)
            snap_flash = 12
            return
        result = pick_best_detection(g_frame, g_boxes, valid)
        if result:
            cropped, label, conf = result
            save_snapshot(cropped, label, conf)
            snap_flash = 12
        else:
            print("  [拍照] 未检测到目标")


def do_exit():
    global running
    running = False


def do_prev_page():
    global sel_page
    if sel_page > 0:
        sel_page -= 1


def do_next_page():
    global sel_page
    max_page = (len(cn_list) - 1) // get_items_per_page()
    if sel_page < max_page:
        sel_page += 1


def get_items_per_page():
    """Calculate how many items fit per page based on screen height."""
    return 60  # default, will be recalculated dynamically


def do_quick_focus(idx):
    name = quick_items[idx]
    set_focus(name)


def set_focus(name):
    global focus, focus_cn, focus_id, inp, sel
    cid = find_cid(name)
    if cid >= 0:
        focus = True
        focus_cn = name
        focus_id = cid
        print(f"-> {name}")
    inp = ""
    sel = False


def build_buttons(w, h):
    """Build clickable buttons based on current screen size and mode."""
    btn_mgr.clear()

    if sel:
        # ====== Selector mode ======
        btn_w, btn_h = 90, 28
        gap = 8
        y = h - btn_h - 8

        # Camera prev (bottom-left area)
        cam_w = 100
        btn_mgr.add(Button(
            10, y, cam_w, btn_h,
            f"◀ 摄像头{cur}", lambda: do_switch_cam(-1),
            color=(40, 80, 140), text_color=(255, 255, 255), font_size=13
        ))
        btn_mgr.add(Button(
            10 + cam_w + gap, y, cam_w, btn_h,
            f"摄像头{cur} ▶", lambda: do_switch_cam(1),
            color=(40, 80, 140), text_color=(255, 255, 255), font_size=13
        ))

        # Page buttons (center)
        btn_mgr.add(Button(
            w // 2 - btn_w - gap // 2, y, btn_w, btn_h,
            "◀ 上一页", do_prev_page,
            color=(80, 80, 160), text_color=(255, 255, 255), font_size=13
        ))
        btn_mgr.add(Button(
            w // 2 + gap // 2, y, btn_w, btn_h,
            "下一页 ▶", do_next_page,
            color=(80, 80, 160), text_color=(255, 255, 255), font_size=13
        ))

        # Exit selector button
        btn_mgr.add(Button(
            w - btn_w - 10, y, btn_w, btn_h,
            "✕ 退出", do_toggle_selector,
            color=(160, 60, 60), text_color=(255, 255, 255), font_size=13
        ))
    else:
        # ====== Normal mode ======
        # 显示当前摄像头名称
        cam_display = cam_names[cur] if cur < len(cam_names) else f"摄像头{cur}"
        btn_w = 85
        btn_h = 28
        gap = 6
        start_x = 10
        y = h - btn_h - 8

        # Quick focus buttons (bottom row, left side)
        for i, item in enumerate(quick_items[:8]):
            is_focused = (focus and focus_cn == item)
            bg = (60, 140, 60) if is_focused else (60, 60, 60)
            btn_mgr.add(Button(
                start_x + i * (btn_w + gap), y, btn_w, btn_h,
                f"{i+1}:{item}", lambda idx=i: do_quick_focus(idx),
                color=bg, text_color=(255, 255, 255), font_size=13
            ))

        # Camera switch buttons (above quick buttons, left side)
        cam_w_btn = 120
        cam_y = y - btn_h - gap
        btn_mgr.add(Button(
            start_x, cam_y, cam_w_btn, btn_h,
            f"◀ {cam_display}", lambda: do_switch_cam(-1),
            color=(40, 80, 140), text_color=(255, 255, 255), font_size=11
        ))
        btn_mgr.add(Button(
            start_x + cam_w_btn + gap, cam_y, cam_w_btn, btn_h,
            f"{cam_display} ▶", lambda: do_switch_cam(1),
            color=(40, 80, 140), text_color=(255, 255, 255), font_size=11
        ))

        # Function buttons on the right side
        fn_w = 70
        fn_x = w - fn_w - 10
        fn_start_y = h - btn_h - 8
        fn_labels = [
            ("模式切换", do_toggle_focus, (60, 60, 140)),
            ("选分类", do_toggle_selector, (60, 60, 140)),
            ("📷 拍照", do_manual_snap, (40, 120, 80)),
            ("自动拍", do_toggle_auto_snap, (120, 100, 40)),
            ("拍全屏", do_toggle_snap_mode, (80, 80, 80)),
            ("全屏", toggle_full, (80, 80, 80)),
            ("暂停", do_toggle_pause, (80, 80, 80)),
            ("帮助", do_toggle_help, (80, 80, 80)),
            ("退出", do_exit, (140, 40, 40)),
        ]
        for i, (label, cb, color) in enumerate(fn_labels):
            is_auto = (label == "自动拍" and auto_snap)
            is_snap_full = (label == "拍全屏" and snap_mode == "full")
            btn_mgr.add(Button(
                fn_x, fn_start_y - (i + 1) * (btn_h + gap), fn_w, btn_h,
                label, cb,
                color=(80, 140, 60) if is_auto else ((60, 120, 140) if is_snap_full else color),
                font_size=12
            ))


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


cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 960, 540)
cv2.setMouseCallback(win, mouse_callback)

print("\n===== Controls =====")
print("  Q:Quit  F:Fullscreen  S:Mode  C/X:Camera  P:Pause")
print("  1-8: Quick select  T:Category list  H:Help on screen")
print("  Mouse: Click buttons | Double-click: Fullscreen")
print("====================\n")

running = True
while running:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = open_cam(cur)
            ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        boxes = results[0].boxes
        g_frame = frame.copy()
        g_boxes = boxes
        ann = frame.copy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            conf = boxes.conf[i].item()
            cid = int(boxes.cls[i].item())
            cn_label = get_cn(all_names[cid])
            color = get_color(cid)
            if focus and cid != focus_id:
                continue
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            ann = draw_cn(ann, f"{cn_label} {conf:.2f}",
                          (x1 + 2, y1 - 22), 16, color, (0, 0, 0))

    h, w = ann.shape[:2]

    if sel:
        # ====== Selector mode ======
        overlay = np.full_like(ann, (20, 20, 20))
        ann = cv2.addWeighted(ann, 0.3, overlay, 0.7, 0)

        ann = draw_cn(ann, "=== 选分类（点击选择 / 输入编号 + 回车确认）===",
                      (20, 15), 20, (0, 255, 0))
        ann = draw_cn(ann, "ESC取消 | 鼠标点击分类项 | 输入编号后回车",
                      (20, 42), 14, (180, 180, 180))

        # Pagination
        per_page = 60  # 3 columns x 20 rows
        start_idx = sel_page * per_page
        end_idx = min(len(cn_list), start_idx + per_page)
        cols = 3
        cw = w // cols

        for i in range(start_idx, end_idx):
            local_i = i - start_idx
            col = local_i % cols
            row = local_i // cols
            x = 15 + col * cw
            y = 70 + row * 25
            txt = f"{i:3d}.{cn_list[i]}"
            c = (255, 255, 0) if (inp and inp.isdigit() and int(inp) == i) else (200, 200, 200)
            ann = draw_cn(ann, txt, (x, y), 14, c)

        if end_idx < len(cn_list):
            max_page = (len(cn_list) - 1) // per_page + 1
            ann = draw_cn(ann,
                          f"...(共{len(cn_list)}项, 第{sel_page+1}/{max_page}页, 点击按钮翻页)",
                          (20, 70 + min(per_page, len(cn_list) - start_idx) * 25 + 5),
                          13, (100, 100, 100))

        # Page info
        max_page = (len(cn_list) - 1) // per_page + 1
        show = inp if inp else "_"
        ann = draw_cn(ann,
                      f">> 编号: {show}  |  第{sel_page+1}/{max_page}页",
                      (20, h - 40), 18, (0, 255, 255), (0, 0, 0))
    else:
        # ====== Normal detection mode ======
        if paused:
            ann = draw_cn(ann, "[已暂停] 按P继续", (w // 2 - 100, 30), 24,
                          (0, 255, 255), (0, 0, 0))

        if focus:
            txt = f"[专注:{focus_cn}]  S:全部  T:选类 M:拍全屏/拍物体"
        else:
            txt = "[全部识别]  S:专注  T:选类"
        ann = draw_cn(ann, txt, (10, 28), 16, (0, 255, 0), (30, 30, 30))

    # ====== Build and draw buttons ======
    build_buttons(w, h)
    ann = btn_mgr.draw_all(ann)

    # Help overlay
    if show_help and not sel:
        help_texts = [
            "===== 帮助 =====",
            "Q:退出 F:全屏 P:暂停 S:模式 T:分类",
            "C/X:切换摄像头 1-8:快捷 H:帮助 M:拍全屏/物体",
            "鼠标:点击按钮 双击:全屏",
            "Space:拍照",
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
    # 分辨率和帧率信息
    res_str = f"{cam_w}x{cam_h} @ {cam_fps}fps"
    info_str = f"{date_str} | {res_str}"
    # Measure text size to right-align
    tmp_pil = Image.new("RGB", (1, 1))
    tmp_d = ImageDraw.Draw(tmp_pil)
    tmp_font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
    bbox = tmp_d.textbbox((0, 0), info_str, font=tmp_font)
    text_w = bbox[2] - bbox[0] + 8
    info_x = w - text_w - 10
    ann = draw_cn(ann, info_str, (info_x, 8), 14, (255, 255, 200))

    # ====== Auto-snapshot ======
    if auto_snap and focus and focus_id >= 0 and not paused and not sel:
        now_time = time.time()
        # Check if target is present
        target_present = False
        best_crop = None
        best_label = ""
        best_conf = 0.0
        best_box = None  # (x1, y1, x2, y2)
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            if cid == focus_id:
                target_present = True
                conf = boxes.conf[i].item()
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    margin = int((x2 - x1) * 0.15)
                    cx1 = max(0, x1 - margin)
                    cy1 = max(0, y1 - margin)
                    cx2 = min(w, x2 + margin)
                    cy2 = min(h, y2 + margin)
                    best_crop = frame[cy1:cy2, cx1:cx2].copy()
                    best_label = get_cn(all_names[cid])
                    best_box = (x1, y1, x2, y2)

        if target_present and best_crop is not None and best_conf >= 0.6:
            # 智能判断是否需要补拍：检查目标是否大幅移动或尺寸大幅变化
            should_snap = False
            if (now_time - last_snap_time) > snap_cooldown:
                if last_snap_target is None:
                    should_snap = True
                else:
                    # 计算目标中心点移动距离（归一化）
                    prev_cx, prev_cy, prev_w, prev_h = last_snap_target
                    curr_cx = (best_box[0] + best_box[2]) / 2
                    curr_cy = (best_box[1] + best_box[3]) / 2
                    curr_w = best_box[2] - best_box[0]
                    curr_h = best_box[3] - best_box[1]

                    # 归一化移动距离
                    move_dist = (((curr_cx - prev_cx) / w) ** 2 + ((curr_cy - prev_cy) / h) ** 2) ** 0.5
                    # 尺寸变化比例
                    prev_area = prev_w * prev_h
                    curr_area = curr_w * curr_h
                    size_change = abs(curr_area - prev_area) / max(prev_area, 1)

                    if move_dist > snap_move_threshold or size_change > snap_size_threshold:
                        should_snap = True

            if should_snap:
                if save_snapshot(best_crop, best_label, best_conf, full_frame=(snap_mode == "full")):
                    last_snap_time = now_time
                    snap_flash = 12
                    # 记录当前目标状态
                    last_snap_target = (
                        (best_box[0] + best_box[2]) / 2,
                        (best_box[1] + best_box[3]) / 2,
                        best_box[2] - best_box[0],
                        best_box[3] - best_box[1]
                    )

    # Flash effect
    if snap_flash > 0:
        ann = cv2.addWeighted(ann, 1.0, np.full_like(ann, 255), 0.4, 0)
        snap_flash -= 1

    # Auto-snap indicator
    if auto_snap:
        mode_txt = f"[自动拍:{focus_cn}|{'全屏' if snap_mode=='full' else '物体'}]" if focus else f"[自动拍:全部|{'全屏' if snap_mode=='full' else '物体'}]"
        ann = draw_cn(ann, mode_txt, (10, 50), 15, (0, 200, 255), (0, 0, 0))


    cv2.imshow(win, ann)

    key = cv2.waitKey(1) & 0xFF

    # ====== Global key handlers ======
    if key == ord("q"):
        break
    elif key == ord("f"):
        toggle_full()
    elif key == ord("c"):
        do_switch_cam(1)
        # 更新摄像头列表
        global cam_names
        cam_names = get_cam_names()
    elif key == ord("x"):
        do_switch_cam(-1)
        # 更新摄像头列表
        global cam_names
        cam_names = get_cam_names()
    elif key == ord("p"):
        do_toggle_pause()
    elif key == ord(" "):  # Space - manual snapshot
        do_manual_snap()
    elif key == ord("s"):
        if sel:
            sel = False
            inp = ""
        else:
            do_toggle_focus()
    elif key == ord("t"):
        do_toggle_selector()
    elif key == ord("h"):
        do_toggle_help()
    elif key == ord("m"):
        do_toggle_snap_mode()
    elif key == 27:  # Esc
        if sel:
            sel = False
            inp = ""
            sel_page = 0
        elif show_help:
            show_help = False

    # ====== Quick keys 1-8 ======
    if not sel:
        for idx in range(min(8, len(quick_items))):
            if key == ord(str(idx + 1)):
                do_quick_focus(idx)
                break

    # ====== Selector mode input ======
    if sel:
        if key == 13 or key == 32:  # Enter or Space
            if inp:
                try:
                    n = int(inp)
                    if 0 <= n < len(cn_list):
                        set_focus(cn_list[n])
                    else:
                        print(f"Invalid number: {n}")
                        inp = ""
                except ValueError:
                    inp = ""
        elif key == 8:  # Backspace
            inp = inp[:-1]
        elif ord("0") <= key <= ord("9"):
            inp += chr(key)

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
