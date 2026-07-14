"""
phone_stream.py
----------------
把 OpenCV 处理后的画面通过 MJPEG 推流到手机浏览器同步观看。

用法（在主程序里）:
    from phone_stream import MJPEGStreamer
    streamer = MJPEGStreamer(port=8000, quality=70)
    streamer.start()                 # 后台线程启动 HTTP 服务
    ...
    streamer.update(annotated_frame)  # 每帧调用，推送最新画面
    ...
    streamer.stop()                   # 退出时释放

手机与电脑连同一 WiFi/热点，手机浏览器打开电脑显示的地址即可实时观看；
若已建立公网隧道（remote_access），手机用流量/异地也能看。
零第三方依赖（仅用标准库 http.server）；qrcode 为可选，仅用于生成二维码。

可选回调（由主程序注入）:
    on_command(action)   -> 手机端按钮触发，action 为 snapshot/record/rec_stop
    file_root / snap_root-> 回放文件目录（录像 / 照片）
    get_file_list()      -> 返回回放列表 [{name,type,url,size}, ...]
    get_status()         -> 返回状态字典（录制中 / 在线数 / 公网地址等）
"""

import os
import threading
import time
import socket
import mimetypes
import collections
import json
from urllib.parse import urlparse, parse_qs, unquote
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def get_lan_ip():
    """获取本机在局域网中的 IP 地址。"""
    ip = "127.0.0.1"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# 手机端网页：整屏自适应显示视频流，深色背景，点击可全屏；
# 底部带「拍照 / 录制」控制；可展开「回放」查看已录视频与照片。
PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,user-scalable=no">
<title>Remote Camera Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}html,body{width:100%;height:100%;background:#05070a;overflow:hidden;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#fff}
#stage{position:fixed;inset:0;width:100vw;height:100vh;height:100dvh;overflow:hidden;background:#10141b;display:flex;align-items:center;justify-content:center}
#ambient{position:absolute;inset:-36px;z-index:0;background-position:center;background-size:cover;filter:blur(28px) brightness(.52) saturate(.85);transform:scale(1.10);opacity:.9}
#v{position:relative;z-index:1;display:block;width:100%;height:100%;max-width:none;max-height:none;object-fit:contain;background:transparent;user-select:none;-webkit-user-select:none;transition:opacity .2s}
body.fit-contain #v{object-fit:contain}.topbar{position:fixed;top:0;left:0;right:0;z-index:8;height:48px;padding:8px max(12px,env(safe-area-inset-left));display:flex;align-items:center;gap:9px;background:linear-gradient(180deg,rgba(0,0,0,.72),transparent);pointer-events:none}.dot{width:9px;height:9px;border-radius:50%;background:#25d981;box-shadow:0 0 10px #25d981}.title{font-size:14px;text-shadow:0 1px 3px #000}.actions{position:fixed;top:8px;right:max(10px,env(safe-area-inset-right));z-index:10;display:flex;gap:7px}.small{border:1px solid rgba(255,255,255,.28);background:rgba(15,20,28,.58);backdrop-filter:blur(10px);color:#fff;border-radius:18px;padding:7px 12px;font-size:13px}.bottom{position:fixed;left:0;right:0;bottom:0;z-index:9;padding:8px max(10px,env(safe-area-inset-right)) calc(10px + env(safe-area-inset-bottom));background:linear-gradient(0deg,rgba(0,0,0,.78),transparent);display:flex;flex-direction:column;gap:7px}.controls{display:grid;grid-template-columns:repeat(6,minmax(74px,1fr));gap:7px;max-width:900px;width:100%;margin:auto}.btn{border:1px solid rgba(255,255,255,.24);background:rgba(20,26,36,.68);color:#fff;border-radius:22px;padding:10px 6px;text-align:center;font-size:14px;white-space:nowrap;backdrop-filter:blur(10px)}.btn.on{background:rgba(15,145,96,.72);border-color:#35e2a3}.btn.rec{background:rgba(190,45,55,.72)}.audio-row{max-width:900px;width:100%;margin:auto;display:flex;align-items:center;gap:8px}.audio-row select{min-width:0;flex:1;border:1px solid rgba(255,255,255,.22);border-radius:16px;padding:7px 10px;background:rgba(20,26,36,.78);color:#fff;font-size:13px}.status{font-size:11px;color:#b9d5ff;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.status a{color:#77d7ff}.panel{position:fixed;inset:0;z-index:20;background:#0b0e14;overflow:auto;display:none;padding:18px max(14px,env(safe-area-inset-left))}.panel h2{font-size:20px;margin-bottom:12px}.close{position:fixed;right:16px;top:14px;color:#9bd5ff}.item{background:#151a23;border-radius:12px;margin-bottom:12px;overflow:hidden}.item video,.item img{width:100%;height:auto;display:block}.cap{padding:7px 10px;color:#bbc;font-size:12px}.hint{color:#8993a4;font-size:12px;margin-bottom:12px}
@media(max-width:680px){.controls{grid-template-columns:repeat(3,1fr)}.controls .secondary{display:none}.btn{font-size:13px;padding:9px 4px}.topbar{height:42px}.title{font-size:12px}.small{padding:6px 10px}.audio-row select{font-size:12px}}
@media(orientation:landscape) and (max-height:560px){.bottom{padding-bottom:6px}.audio-row{max-width:760px}.controls{max-width:760px}.btn{padding:7px 5px}.status{display:none}}
:fullscreen .topbar,:fullscreen .bottom,:fullscreen .actions{opacity:.92}
</style></head><body>
<div id="stage"><div id="ambient"></div><img id="v" src="/stream.mjpg?layout=tile" alt="camera stream"></div>
<div class="topbar"><span class="dot"></span><span class="title">&#20844;&#32593;&#23454;&#26102;&#30417;&#25511; &middot; &#30011;&#38754;&#19982;&#22768;&#38899;</span></div>
<div class="actions"><button class="small" id="fitBtn" onclick="toggleFit()">&#33258;&#36866;&#24212;</button><button class="small" onclick="fs()">&#20840;&#23631;</button></div>
<div class="bottom">
 <div class="audio-row"><button class="btn" id="soundBtn" onclick="toggleSound()">&#24320;&#21551;&#22768;&#38899;</button><select id="audioSelect" onchange="changeAudio()"><option>&#27491;&#22312;&#35835;&#21462;&#40614;&#20811;&#39118;...</option></select></div>
 <div class="controls"><button class="btn" id="modeBtn" onclick="toggleMode()">&#20999;&#21040;&#30417;&#25511;</button><button class="btn" id="camBtn" onclick="toggleMulti()">&#22810;&#25668;&#21516;&#23631;</button><button class="btn" id="autoBtn" onclick="toggleAuto()">&#33258;&#21160;&#25293;</button><button class="btn" id="snapBtn" onclick="snap()">&#25293;&#29031;</button><button class="btn" id="recBtn" onclick="rec()">&#24405;&#21046;</button><button class="btn" id="playbackBtn" onclick="openPanel()">&#22238;&#25918;</button><button class="btn secondary" onclick="toggleFit()">&#23436;&#25972;&#33258;&#36866;&#24212;</button></div>
 <div class="status" id="urlbar">&#27491;&#22312;&#36830;&#25509;&#20844;&#32593;...</div>
</div>
<div class="panel" id="panel"><div class="close" onclick="closePanel()">&#20851;&#38381; &times;</div><h2>&#22238;&#25918;</h2><div class="hint">&#24405;&#20687;&#19982;&#29031;&#29255;&#20445;&#23384;&#22312;&#30005;&#33041;&#31471;&#12290;</div><div id="list"></div></div>
<script>
const img=document.getElementById('v'),ambient=document.getElementById('ambient'),ambientCanvas=document.createElement('canvas');ambientCanvas.width=96;ambientCanvas.height=54;let layout='tile',fit='contain',soundOn=false,audioAbort=null,audioCtx=null,nextAudioTime=0,audioRate=16000;
function wantedLayout(){return innerWidth/Math.max(innerHeight,1)>1.05?'tile':'stack'}
function applyLayout(){const w=wantedLayout();if(w!==layout){layout=w;img.src='/stream.mjpg?layout='+layout+'&vw='+innerWidth+'&vh='+innerHeight+'&t='+Date.now()}applyFit()}
function applyFit(){document.body.classList.add('fit-contain');document.getElementById('fitBtn').textContent='\u81ea\u9002\u5e94'}
function toggleFit(){fit='contain';applyFit()}
function updateAmbient(){try{if(!img.naturalWidth||!img.naturalHeight)return;const c=ambientCanvas.getContext('2d');c.drawImage(img,0,0,ambientCanvas.width,ambientCanvas.height);ambient.style.backgroundImage='url('+ambientCanvas.toDataURL('image/jpeg',.45)+')'}catch(e){}}
function fs(){const e=document.documentElement;if(document.fullscreenElement)document.exitFullscreen();else if(e.requestFullscreen)e.requestFullscreen();else if(e.webkitRequestFullscreen)e.webkitRequestFullscreen()}
img.onerror=()=>setTimeout(()=>img.src='/stream.mjpg?layout='+layout+'&vw='+innerWidth+'&vh='+innerHeight+'&t='+Date.now(),800);img.onload=updateAmbient;setInterval(updateAmbient,900);addEventListener('resize',applyLayout);addEventListener('orientationchange',()=>setTimeout(applyLayout,250));document.addEventListener('fullscreenchange',applyLayout);applyLayout();
async function loadAudioDevices(){try{const d=await fetch('/audio/devices').then(r=>r.json()),sel=document.getElementById('audioSelect');sel.innerHTML='';(d.devices||[]).forEach(x=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.label;if(x.id===d.selected)o.selected=true;sel.appendChild(o)});if(!d.available){sel.innerHTML='<option>'+('\u672a\u627e\u5230\u53ef\u7528\u9ea6\u514b\u98ce')+'</option>';document.getElementById('soundBtn').disabled=true}}catch(e){}}
function joinBytes(a,b){const z=new Uint8Array(a.length+b.length);z.set(a);z.set(b,a.length);return z}
function findHeaderEnd(a){for(let i=0;i+3<a.length;i++)if(a[i]===13&&a[i+1]===10&&a[i+2]===13&&a[i+3]===10)return i;return -1}
function schedulePCM(a){if(a.length%2)a=a.slice(0,-1);if(!a.length)return;const dv=new DataView(a.buffer,a.byteOffset,a.byteLength),n=a.byteLength/2,b=audioCtx.createBuffer(1,n,audioRate),ch=b.getChannelData(0);for(let i=0;i<n;i++)ch[i]=dv.getInt16(i*2,true)/32768;const src=audioCtx.createBufferSource();src.buffer=b;src.connect(audioCtx.destination);if(nextAudioTime<audioCtx.currentTime+.04)nextAudioTime=audioCtx.currentTime+.08;if(nextAudioTime>audioCtx.currentTime+.8)nextAudioTime=audioCtx.currentTime+.12;src.start(nextAudioTime);nextAudioTime+=n/audioRate}
async function startSound(){if(soundOn)return;const btn=document.getElementById('soundBtn');try{audioCtx=audioCtx||new(window.AudioContext||window.webkitAudioContext)({sampleRate:audioRate});await audioCtx.resume();audioAbort=new AbortController();soundOn=true;btn.textContent='\u5173\u95ed\u58f0\u97f3';btn.classList.add('on');nextAudioTime=audioCtx.currentTime+.10;let seq=0;while(soundOn){const res=await fetch('/audio/chunk?after='+seq+'&t='+Date.now(),{signal:audioAbort.signal,cache:'no-store'});if(!res.ok)throw new Error(await res.text());audioRate=parseInt(res.headers.get('X-Sample-Rate')||'16000');seq=parseInt(res.headers.get('X-Audio-Seq')||seq);const pcm=new Uint8Array(await res.arrayBuffer());if(soundOn&&pcm.length)schedulePCM(pcm)}}catch(e){if(soundOn)console.log(e)}finally{if(soundOn)stopSound()}}
function stopSound(){soundOn=false;if(audioAbort)audioAbort.abort();audioAbort=null;const b=document.getElementById('soundBtn');b.textContent='\u5f00\u542f\u58f0\u97f3';b.classList.remove('on')}
function toggleSound(){soundOn?stopSound():startSound()}
async function changeAudio(){const id=document.getElementById('audioSelect').value,was=soundOn;stopSound();await fetch('/audio/select?device='+encodeURIComponent(id));if(was)setTimeout(startSound,250)}
function refresh(){fetch('/status').then(r=>r.json()).then(s=>{const u=s.remote||s.lan||'',fps=typeof s.stream_fps==='number'?' &middot; '+s.stream_fps.toFixed(1)+' FPS':'',aud=s.audio_level===undefined?'':' &middot; '+(s.audio_level>0.003?'\u58f0\u97f3\u6b63\u5e38':'\u7b49\u5f85\u58f0\u97f3');document.getElementById('urlbar').innerHTML=(s.remote?'\u516c\u7f51':'\u5c40\u57df\u7f51')+'\u5730\u5740: <a href="'+u+'">'+u+'</a>'+fps+aud;const b=document.getElementById('modeBtn'),on=s.detection_enabled!==false;b.textContent=on?'\u5207\u5230\u76d1\u63a7':'\u5f00\u542f\u8bc6\u522b';b.className='btn'+(on?'':' on');const cb=document.getElementById('camBtn');if(cb){cb.disabled=!!s.camera_switching;cb.textContent=s.camera_switching?'\u5207\u6362\u4e2d...':(s.multi_camera?'\u5207\u56de\u5355\u6444':'\u591a\u6444\u540c\u5c4f');cb.className='btn'+(s.multi_camera?' on':'')}const ab=document.getElementById('autoBtn');if(ab){ab.textContent=s.auto_snap?'\u81ea\u52a8\u62cd\u5df2\u5f00':'\u81ea\u52a8\u62cd';ab.className='btn'+(s.auto_snap?' on':'')}}).catch(()=>{})}setInterval(refresh,3000);refresh();loadAudioDevices();
function toggleMode(){fetch('/cmd?action=detection_toggle').then(()=>setTimeout(refresh,120))}function toggleAuto(){fetch('/cmd?action=auto_snap_toggle').then(()=>setTimeout(refresh,180))}function toggleMulti(){const b=document.getElementById('camBtn');b.textContent='\u5207\u6362\u4e2d...';fetch('/cmd?action=multi_toggle').then(()=>setTimeout(refresh,2200))}function snap(){fetch('/cmd?action=snapshot').then(()=>{const b=document.getElementById('snapBtn');b.textContent='\u5df2\u62cd\u7167\u2713';setTimeout(()=>b.textContent='\u62cd\u7167',900)})}function rec(){fetch('/cmd?action=record').then(r=>r.text()).then(t=>{const b=document.getElementById('recBtn'),on=t.includes('on');b.textContent=on?'\u505c\u6b62\u5f55\u5236':'\u5f55\u5236';b.className='btn'+(on?' rec':'')})}
function openPanel(){document.getElementById('panel').style.display='block';loadList()}function closePanel(){document.getElementById('panel').style.display='none'}function loadList(){fetch('/files').then(r=>r.json()).then(items=>{const b=document.getElementById('list');b.innerHTML='';if(!items.length){b.innerHTML='<div class="hint">No files</div>';return}items.forEach(it=>{const d=document.createElement('div');d.className='item';d.innerHTML=(it.type==='video'?'<video src="'+it.url+'" controls></video>':'<img src="'+it.url+'">')+'<div class="cap">'+it.name+'</div>';b.appendChild(d)})})}
</script></body></html>"""


class AudioCapture:
    """Low-latency mono PCM microphone capture shared by all remote viewers."""
    def __init__(self, sample_rate=16000, block_ms=40):
        self.sample_rate = int(sample_rate)
        self.blocksize = max(160, int(self.sample_rate * block_ms / 1000))
        self._sd = None
        self._devices = []
        self.selected = None
        self._stream = None
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._chunks = collections.deque(maxlen=250)
        self._seq = 0
        self._clients = 0
        self.level = 0.0
        self.error = None
        self._discover()

    def _discover(self):
        try:
            import sounddevice as sd
            self._sd = sd
            hostapis = sd.query_hostapis()
            devices = sd.query_devices()
            result = []
            for i, dev in enumerate(devices):
                if int(dev.get('max_input_channels', 0)) < 1:
                    continue
                host = hostapis[int(dev['hostapi'])]['name']
                # MME gives one stable, human-readable entry per physical input.
                if host != 'MME':
                    continue
                name = str(dev['name'])
                low = name.lower()
                if i == 0 or 'mapper' in low or 'streaming' in low or 'virtual' in low or (chr(0x7F51) + chr(0x6613)) in name:
                    continue
                result.append({'id': i, 'label': name, 'hostapi': host})
            if not result:
                for i, dev in enumerate(devices):
                    if int(dev.get('max_input_channels', 0)) > 0:
                        result.append({'id': i, 'label': str(dev['name']), 'hostapi': hostapis[int(dev['hostapi'])]['name']})
            self._devices = result
            preferred = next((d for d in result if 'camera' in d['label'].lower()), None)
            if preferred is None:
                preferred = next((d for d in result if 'usb audio' in d['label'].lower()), None)
            self.selected = preferred['id'] if preferred else (result[0]['id'] if result else None)
        except Exception as exc:
            self.error = str(exc)
            self._devices = []
            self.selected = None

    def devices_payload(self):
        return {'available': bool(self._devices), 'devices': list(self._devices),
                'selected': self.selected, 'sample_rate': self.sample_rate,
                'error': self.error}

    def _callback(self, indata, frames, time_info, status):
        data = bytes(indata)
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            level = float(np.mean(np.abs(samples.astype(np.float32))) / 32768.0) if samples.size else 0.0
        except Exception:
            level = 0.0
        with self._cond:
            self.level = self.level * 0.82 + level * 0.18
            self._seq += 1
            self._chunks.append((self._seq, data))
            self._cond.notify_all()

    def start(self):
        with self._lock:
            if self._stream is not None:
                return True
            if self._sd is None or self.selected is None:
                self.error = self.error or 'No microphone input is available'
                return False
            try:
                self._chunks.clear()
                self._seq = 0
                self.level = 0.0
                self._stream = self._sd.RawInputStream(
                    samplerate=self.sample_rate, blocksize=self.blocksize,
                    device=int(self.selected), channels=1, dtype='int16',
                    callback=self._callback, latency='low')
                self._stream.start()
                self.error = None
                print(f"[Audio] microphone {self.selected} started at {self.sample_rate} Hz")
                return True
            except Exception as exc:
                self.error = str(exc)
                self._stream = None
                print(f"[Audio] start failed: {exc}")
                return False

    def stop(self):
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def select(self, device_id):
        try:
            device_id = int(device_id)
        except Exception:
            return False
        if not any(int(d['id']) == device_id for d in self._devices):
            return False
        with self._lock:
            changed = device_id != self.selected
            self.selected = device_id
            active = self._clients > 0
        if changed:
            self.stop()
            if active:
                self.start()
        return True

    def client_inc(self):
        with self._lock:
            self._clients += 1
        return self.start()

    def client_dec(self):
        with self._lock:
            self._clients = max(0, self._clients - 1)
            should_stop = self._clients == 0
        if should_stop:
            self.stop()

    def wait_chunk(self, last_seq, timeout=2.0):
        with self._cond:
            deadline = time.monotonic() + timeout
            while self._seq <= last_seq and time.monotonic() < deadline:
                self._cond.wait(max(0.01, deadline - time.monotonic()))
            if not self._chunks:
                return None, last_seq
            for seq, data in self._chunks:
                if seq > last_seq:
                    return data, seq
            return self._chunks[-1][1], self._chunks[-1][0]

    def read_since(self, last_seq, target_chunks=4, max_chunks=16, timeout=0.8):
        """Return a short, recent PCM batch for cache-free HTTP long polling."""
        try:
            last_seq = max(0, int(last_seq))
        except Exception:
            last_seq = 0
        if not self.start():
            return None, last_seq
        if last_seq == 0:
            target_chunks = max(int(target_chunks), 12)
        with self._cond:
            deadline = time.monotonic() + timeout
            target_seq = last_seq + max(1, int(target_chunks))
            while self._seq < target_seq and time.monotonic() < deadline:
                self._cond.wait(max(0.01, deadline - time.monotonic()))
            fresh = [(seq, data) for seq, data in self._chunks if seq > last_seq]
            if not fresh:
                return b"", self._seq
            fresh = fresh[-max(1, int(max_chunks)):]
            return b"".join(data for _, data in fresh), fresh[-1][0]

    def status(self):
        with self._lock:
            return {'audio_available': bool(self._devices), 'audio_device': self.selected,
                    'audio_clients': self._clients, 'audio_level': self.level,
                    'audio_sample_rate': self.sample_rate, 'audio_error': self.error}


class MJPEGStreamer:
    def __init__(self, port=8000, quality=68, fps_cap=30,
                 max_width=960, max_height=720,
                 on_command=None, file_root=None, snap_root=None,
                 get_file_list=None, get_status=None):
        self.port = port
        self.quality = int(quality)
        self.fps_cap = max(int(fps_cap), 1)
        self.min_interval = 1.0 / self.fps_cap
        # Camera frame intervals always have a few milliseconds of jitter.  A strict
        # 1/fps comparison drops almost every second 30 FPS frame when it arrives
        # slightly early (for example at 31 ms instead of 33.3 ms).  Allow a small
        # scheduling tolerance while still rejecting genuinely excessive rates.
        self.submit_interval = self.min_interval * 0.82
        self.max_width = max(int(max_width), 0)
        self.max_height = max(int(max_height), 0)
        self._jpeg = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._last_submit = 0.0
        self._clients = 0
        self._clients_lock = threading.Lock()
        self.phone_layout = "stack"
        self._server = None
        self._thread = None
        self._running = True
        self._pending_frame = None
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._encoded_times = []
        self.encoded_fps = 0.0
        self.encode_ms = 0.0
        self.jpeg_kb = 0.0
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, name="mjpeg-encoder", daemon=True)
        self._encoder_thread.start()
        self.lan_ip = get_lan_ip()
        self.on_command = on_command
        self.file_root = file_root
        self.snap_root = snap_root
        self.get_file_list = get_file_list
        self.get_status = get_status
        self.audio = AudioCapture()

    # ---------- asynchronous frame update ----------
    def has_clients(self):
        with self._clients_lock:
            return self._clients > 0

    def client_count(self):
        """Return the number of connected viewers."""
        with self._clients_lock:
            return self._clients

    def update(self, frame):
        """Submit the newest BGR frame without blocking on JPEG encoding.

        Only the newest pending frame is retained. Slow clients therefore lose
        old frames instead of accumulating visible delay.
        """
        if frame is None or not self.has_clients():
            return False
        now = time.perf_counter()
        if now - self._last_submit < self.submit_interval:
            return False
        self._last_submit = now
        try:
            latest = frame.copy()
        except Exception:
            return False
        with self._pending_lock:
            self._pending_frame = latest
        self._pending_event.set()
        return True

    def _fit_stream_size(self, frame):
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return frame
        scale = 1.0
        if self.max_width and w > self.max_width:
            scale = min(scale, self.max_width / w)
        if self.max_height and h > self.max_height:
            scale = min(scale, self.max_height / h)
        if scale >= 0.999:
            return frame
        nw = max(2, int(w * scale) // 2 * 2)
        nh = max(2, int(h * scale) // 2 * 2)
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _encoder_loop(self):
        while self._running:
            self._pending_event.wait(0.5)
            self._pending_event.clear()
            if not self._running:
                break
            with self._pending_lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None or not self.has_clients():
                continue
            t0 = time.perf_counter()
            try:
                frame = self._fit_stream_size(frame)
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality,
                     cv2.IMWRITE_JPEG_OPTIMIZE, 0])
            except Exception:
                continue
            if not ok:
                continue
            data = buf.tobytes()
            done = time.perf_counter()
            self.encode_ms = (done - t0) * 1000.0
            self.jpeg_kb = len(data) / 1024.0
            self._encoded_times.append(done)
            cutoff = done - 2.0
            while self._encoded_times and self._encoded_times[0] < cutoff:
                self._encoded_times.pop(0)
            if len(self._encoded_times) >= 2:
                dt = self._encoded_times[-1] - self._encoded_times[0]
                self.encoded_fps = (len(self._encoded_times) - 1) / max(dt, 1e-6)
            with self._cond:
                self._jpeg = data
                self._seq += 1
                self._cond.notify_all()

    def _wait_frame(self, last_seq, timeout=5.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def _client_inc(self):
        with self._clients_lock:
            self._clients += 1

    def _client_dec(self):
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)

    # ---------- 安全取文件 ----------
    def _safe_path(self, root, name):
        if root is None or not name:
            return None
        # 浏览器会把文件名中的中文/空格做百分号编码（如 全屏 -> %E5%85%A8%E5%B1%8F），
        # 必须先解码才能匹配磁盘上的真实文件名，否则中文名快照会 404 无法显示。
        name = unquote(name)
        base = os.path.basename(name)  # 防目录穿越
        p = os.path.abspath(os.path.join(root, base))
        if os.path.dirname(p) != os.path.abspath(root):
            return None
        return p if os.path.isfile(p) else None

    # ---------- 服务启动/停止 ----------
    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass  # 静默日志

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send_page()
                elif path == "/stream.mjpg":
                    self._send_stream()
                elif path == "/audio.pcm":
                    self._send_audio()
                elif path == "/audio/chunk":
                    self._send_audio_chunk()
                elif path == "/audio/devices":
                    self._send_audio_devices()
                elif path == "/audio/select":
                    self._select_audio()
                elif path == "/cmd":
                    self._send_cmd()
                elif path == "/files":
                    self._send_files()
                elif path == "/status":
                    self._send_status()
                elif path.startswith("/file/"):
                    self._send_file(streamer.file_root, path[len("/file/"):])
                elif path.startswith("/snap/"):
                    self._send_file(streamer.snap_root, path[len("/snap/"):])
                else:
                    self.send_error(404)

            def _send_audio_devices(self):
                body = json.dumps(streamer.audio.devices_payload(), ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _select_audio(self):
                q = parse_qs(urlparse(self.path).query)
                ok = streamer.audio.select(q.get('device', [''])[0])
                body = json.dumps({'ok': ok, **streamer.audio.devices_payload()}, ensure_ascii=False).encode('utf-8')
                self.send_response(200 if ok else 400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _send_audio_chunk(self):
                q = parse_qs(urlparse(self.path).query)
                data, seq = streamer.audio.read_since(q.get("after", ["0"])[0])
                if data is None:
                    body = (streamer.audio.error or "audio unavailable").encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache, no-store, no-transform")
                self.send_header("Pragma", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("X-Sample-Rate", str(streamer.audio.sample_rate))
                self.send_header("X-Audio-Seq", str(seq))
                self.end_headers()
                self.wfile.write(data)

            def _send_audio(self):
                if not streamer.audio.client_inc():
                    streamer.audio.client_dec()
                    body = (streamer.audio.error or 'audio unavailable').encode('utf-8')
                    self.send_response(503)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=audio')
                self.send_header('Cache-Control', 'no-cache, no-store, no-transform')
                self.send_header('Pragma', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('X-Sample-Rate', str(streamer.audio.sample_rate))
                self.send_header('X-Audio-Format', 's16le-mono')
                self.end_headers()
                last_seq = 0
                pending = bytearray()
                # Cloudflare buffers very small audio writes. One ~1 second PCM
                # part crosses its streaming threshold and avoids 4-5 second bursts.
                part_bytes = streamer.audio.sample_rate * 2
                try:
                    while True:
                        data, last_seq = streamer.audio.wait_chunk(last_seq)
                        if data:
                            pending.extend(data)
                        if len(pending) >= part_bytes:
                            payload = bytes(pending)
                            pending.clear()
                            self.wfile.write(b'--audio\r\n')
                            self.wfile.write(b'Content-Type: application/octet-stream\r\n')
                            self.wfile.write(('Content-Length: %d\r\n\r\n' % len(payload)).encode('ascii'))
                            self.wfile.write(payload)
                            self.wfile.write(b'\r\n')
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    streamer.audio.client_dec()

            def _send_page(self):
                body = PAGE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def _send_stream(self):
                # 手机端可带 ?layout=tile|stack 切换多摄布局（横屏平铺/竖屏竖排）
                q = parse_qs(urlparse(self.path).query)
                ly = q.get("layout", ["stack"])[0]
                if ly in ("stack", "tile"):
                    streamer.phone_layout = ly
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, no-store, no-transform, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                streamer._client_inc()
                last_seq = -1
                try:
                    while True:
                        jpeg, last_seq = streamer._wait_frame(last_seq)
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(jpeg)).encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                except Exception:
                    pass
                finally:
                    streamer._client_dec()

            def _send_cmd(self):
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                action = q.get("action", [""])[0]
                try:
                    if streamer.on_command is not None:
                        msg = streamer.on_command(action)
                    else:
                        msg = "no handler"
                except Exception as e:
                    msg = "error: %s" % e
                body = (msg or "ok").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_files(self):
                try:
                    items = streamer.get_file_list() if streamer.get_file_list else []
                except Exception:
                    items = []
                import json as _json
                body = _json.dumps(items, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_status(self):
                import json as _json
                try:
                    st = streamer.get_status() if streamer.get_status else {}
                except Exception:
                    st = {}
                st.update(streamer.audio.status())
                body = _json.dumps(st, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_file(self, root, name):
                p = streamer._safe_path(root, name)
                if p is None:
                    self.send_error(404)
                    return
                ctype, _ = mimetypes.guess_type(p)
                ctype = ctype or "application/octet-stream"
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except Exception:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)

        try:
            self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as e:
            print(f"[推流] 端口 {self.port} 启动失败: {e}")
            return False
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print("\n" + "=" * 46)
        print("  手机同步观看已开启")
        print(f"  同一WiFi:  http://{self.lan_ip}:{self.port}")
        print("  免WiFi远程: 见公网地址（建立隧道后显示）")
        print("=" * 46 + "\n")
        return True

    def stop(self):
        self._running = False
        self.audio.stop()
        self._pending_event.set()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._encoder_thread is not None and self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=2.0)

    @property
    def url(self):
        return f"http://{self.lan_ip}:{self.port}"


def make_qr_image(text, box_size=6, border=2):
    """生成二维码的 OpenCV BGR 图像；qrcode 未安装则返回 None。"""
    try:
        import qrcode
    except Exception:
        return None
    try:
        qr = qrcode.QRCode(box_size=box_size, border=border,
                           error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        arr = np.array(img)  # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None
