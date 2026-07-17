"""Low-latency MJPEG and PCM web monitor used by yolo_cam.py."""

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
    """Return the LAN IPv4 address used by remote viewers."""
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


# 鎵嬫満绔綉椤碉細鏁村睆鑷€傚簲鏄剧ず瑙嗛娴侊紝娣辫壊鑳屾櫙锛岀偣鍑诲彲鍏ㄥ睆锛?
# 搴曢儴甯︺€屾媿鐓?/ 褰曞埗銆嶆帶鍒讹紱鍙睍寮€銆屽洖鏀俱€嶆煡鐪嬪凡褰曡棰戜笌鐓х墖銆?
PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,user-scalable=no"><meta name="theme-color" content="#050913"><title>NEURAL WATCH &middot; &#26234;&#33021;&#36828;&#31243;&#30417;&#25511;</title>
<style>
:root{--cyan:#42e8ff;--blue:#4d7cff;--violet:#a56cff;--green:#42f5ad;--red:#ff5470;--ink:#050913;--panel:rgba(8,16,31,.82);--line:rgba(99,221,255,.25);--muted:#8198b5}*{box-sizing:border-box;margin:0;padding:0}html,body{width:100%;height:100%;overflow:hidden;background:var(--ink);color:#eefaff;font-family:"Segoe UI","Microsoft YaHei",sans-serif;-webkit-tap-highlight-color:transparent}body:before{content:"";position:fixed;inset:0;z-index:0;background:radial-gradient(circle at 15% 15%,rgba(61,96,255,.18),transparent 32%),radial-gradient(circle at 85% 80%,rgba(0,226,255,.12),transparent 34%),linear-gradient(rgba(50,160,210,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(50,160,210,.035) 1px,transparent 1px);background-size:auto,auto,32px 32px,32px 32px;pointer-events:none}#stage{position:fixed;inset:0;z-index:1;display:flex;align-items:center;justify-content:center;padding:58px 10px 188px;background:radial-gradient(ellipse at center,rgba(12,26,45,.72),rgba(2,6,13,.98))}#stage:after{content:"";position:absolute;inset:58px 10px 188px;border:1px solid rgba(66,232,255,.16);border-radius:18px;box-shadow:inset 0 0 35px rgba(20,90,140,.12);pointer-events:none}#v{position:relative;z-index:2;display:block;width:auto;height:auto;max-width:100%;max-height:100%;object-fit:contain;border-radius:16px;background:transparent;filter:saturate(1.03) contrast(1.02);user-select:none;-webkit-user-select:none}body.fullscreen-view #stage{padding:0}body.fullscreen-view #stage:after{inset:0;border-radius:0}body.fullscreen-view .hud,body.fullscreen-view .console{display:none}body.fullscreen-view #v{max-width:100vw;max-height:100vh;border-radius:0}.scan{position:fixed;inset:0;z-index:3;pointer-events:none;opacity:.12;background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(105,230,255,.08) 4px)}.hud{position:fixed;left:0;right:0;top:0;z-index:8;height:58px;padding:9px max(12px,env(safe-area-inset-left));display:flex;align-items:center;gap:14px;background:linear-gradient(180deg,rgba(3,8,18,.98),rgba(3,8,18,.72),transparent);backdrop-filter:blur(8px)}.brand{display:flex;align-items:center;gap:10px;min-width:190px}.brand-mark{width:32px;height:32px;border:1px solid var(--cyan);border-radius:9px;display:grid;place-items:center;box-shadow:0 0 18px rgba(66,232,255,.28);font-weight:800;color:var(--cyan)}.brand-main{font-size:14px;font-weight:800;letter-spacing:1.8px}.brand-sub{font-size:9px;color:#6f91b1;letter-spacing:1.2px;margin-top:2px}.live{display:flex;align-items:center;gap:6px;font-size:11px;color:#b7cbe0}.live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);animation:pulse 1.7s infinite}@keyframes pulse{50%{opacity:.35}}.metrics{display:flex;gap:7px;flex:1;justify-content:center}.chip{min-width:78px;padding:6px 9px;border:1px solid var(--line);border-radius:9px;background:rgba(6,18,34,.7);text-align:center}.chip b{display:block;font-size:12px;color:#eafbff}.chip span{display:block;font-size:8px;color:#6484a2;letter-spacing:.8px;margin-top:1px}.head-actions{display:flex;gap:7px}.icon-btn{height:34px;border:1px solid var(--line);border-radius:10px;padding:0 11px;background:rgba(8,19,36,.78);color:#dffaff;font-size:12px;cursor:pointer}.icon-btn:active{transform:scale(.96)}.console{position:fixed;z-index:9;left:10px;right:10px;bottom:calc(9px + env(safe-area-inset-bottom));max-width:1180px;margin:auto;padding:10px;border:1px solid rgba(66,232,255,.28);border-radius:18px;background:linear-gradient(145deg,rgba(7,15,29,.92),rgba(8,19,36,.82));box-shadow:0 18px 60px rgba(0,0,0,.5),inset 0 0 30px rgba(35,135,190,.05);backdrop-filter:blur(18px)}.console:before,.console:after{content:"";position:absolute;width:22px;height:22px;border-color:var(--cyan);opacity:.65}.console:before{left:-1px;top:-1px;border-left:2px solid;border-top:2px solid;border-radius:18px 0 0}.console:after{right:-1px;bottom:-1px;border-right:2px solid;border-bottom:2px solid;border-radius:0 0 18px}.console-top{display:flex;align-items:center;gap:9px;margin-bottom:9px}.address{min-width:0;flex:1;height:34px;border:1px solid rgba(95,186,225,.2);border-radius:10px;background:rgba(2,9,18,.7);display:flex;align-items:center;padding:0 10px;gap:8px}.address-label{font-size:9px;color:var(--cyan);letter-spacing:1px;white-space:nowrap}.address a{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#d8f7ff;text-decoration:none;font-size:11px}.copy-state{font-size:9px;color:var(--green);white-space:nowrap}.profile{display:flex;border:1px solid rgba(95,186,225,.2);border-radius:10px;overflow:hidden;background:rgba(2,9,18,.7)}.profile button{height:32px;padding:0 10px;border:0;border-right:1px solid rgba(95,186,225,.16);background:transparent;color:#7893ad;font-size:10px}.profile button:last-child{border-right:0}.profile button.on{background:linear-gradient(135deg,rgba(31,122,180,.5),rgba(67,72,190,.45));color:#fff;box-shadow:inset 0 -2px var(--cyan)}.controls{display:grid;grid-template-columns:repeat(9,1fr);gap:7px}.ctrl{height:52px;border:1px solid rgba(87,177,215,.22);border-radius:12px;background:linear-gradient(145deg,rgba(15,31,52,.86),rgba(7,15,28,.92));color:#d9edff;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:3px;font-size:11px;cursor:pointer;transition:.16s}.ctrl .ico{font-size:16px;color:#78ddff;text-shadow:0 0 12px rgba(66,232,255,.45)}.ctrl:hover{border-color:rgba(66,232,255,.55);transform:translateY(-1px)}.ctrl:active{transform:scale(.96)}.ctrl.on{border-color:rgba(66,245,173,.72);background:linear-gradient(145deg,rgba(15,99,82,.72),rgba(7,35,39,.92))}.ctrl.on .ico{color:var(--green)}.ctrl.rec{border-color:rgba(255,84,112,.75);background:rgba(100,20,38,.75)}.ctrl:disabled{opacity:.5}.audio-row{display:flex;gap:7px;align-items:center;margin-top:8px}.sound{height:32px;min-width:100px}.audio-row select{height:32px;min-width:0;flex:1;border:1px solid rgba(95,186,225,.2);border-radius:9px;background:#071224;color:#bcd4e8;padding:0 9px;font-size:10px}.stream-note{font-size:9px;color:#5d7791;white-space:nowrap}.drawer{position:fixed;inset:0;z-index:30;display:none;background:rgba(2,6,13,.82);backdrop-filter:blur(14px);padding:18px;overflow:auto}.drawer-card{max-width:920px;margin:auto;border:1px solid rgba(66,232,255,.28);border-radius:20px;background:#07101f;padding:18px;box-shadow:0 30px 80px #000}.drawer-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:15px}.drawer h2{font-size:18px;letter-spacing:1px}.drawer-close{border:1px solid var(--line);border-radius:9px;background:#0d1b31;color:#cceeff;padding:7px 12px}.cam-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.cam-item{border:1px solid rgba(90,170,210,.24);border-radius:13px;background:#0b192c;color:#c8def0;padding:15px;text-align:left}.cam-item.on{border-color:var(--green);box-shadow:inset 3px 0 var(--green);color:#fff}.file-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.file-item{border:1px solid rgba(90,170,210,.18);border-radius:13px;overflow:hidden;background:#091526}.file-item img,.file-item video{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#02050a}.file-cap{padding:8px;color:#8fa8bf;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.empty{padding:30px;text-align:center;color:#6f879f}.collapse{display:none}@media(max-width:760px){#stage{padding:50px 5px 220px}#stage:after{inset:50px 5px 220px}.hud{height:50px;padding:7px 9px}.brand{min-width:auto}.brand-sub,.metrics .chip:nth-child(n+3){display:none}.metrics{justify-content:flex-end}.chip{min-width:58px;padding:5px}.head-actions .icon-btn:first-child{display:none}.console{left:5px;right:5px;bottom:calc(5px + env(safe-area-inset-bottom));padding:8px;border-radius:14px}.console-top{flex-wrap:wrap;margin-bottom:7px}.address{order:1;width:100%;flex-basis:100%}.profile{order:2;flex:1}.collapse{display:block;order:3}.controls{grid-template-columns:repeat(4,1fr)}.ctrl{height:46px;font-size:10px}.ctrl .ico{font-size:14px}.audio-row{margin-top:7px}.stream-note{display:none}.console.compact .controls,.console.compact .audio-row,.console.compact .profile{display:none}.console.compact{padding-bottom:8px}}@media(orientation:landscape) and (max-height:600px){#stage{padding:46px 220px 5px 5px}#stage:after{inset:46px 220px 5px 5px}.hud{height:46px}.metrics .chip:nth-child(n+2){display:none}.console{left:auto;right:5px;top:50px;bottom:5px;width:210px;overflow:auto}.console-top{display:block}.address{margin-bottom:7px}.profile{margin-bottom:7px}.controls{grid-template-columns:repeat(2,1fr)}.ctrl{height:43px}.audio-row{display:block}.sound,.audio-row select{width:100%;margin-top:6px}.stream-note{display:none}}
</style></head><body>
<div id="stage"><img id="v" src="/stream.mjpg?layout=tile" alt="&#23454;&#26102;&#25668;&#20687;&#22836;&#30011;&#38754;"></div><div class="scan"></div>
<header class="hud"><div class="brand"><div class="brand-mark">N</div><div><div class="brand-main">NEURAL WATCH</div><div class="brand-sub">AI VISUAL MONITORING SYSTEM</div></div></div><div class="live"><i class="live-dot"></i><span id="liveText">LIVE</span></div><div class="metrics"><div class="chip"><b id="modeMetric">AI</b><span>AI MODE</span></div><div class="chip"><b id="fpsMetric">-- FPS</b><span>STREAM</span></div><div class="chip"><b id="camMetric">-- CAM</b><span>CAMERAS</span></div><div class="chip"><b id="netMetric">-- ms</b><span>NETWORK</span></div></div><div class="head-actions"><button class="icon-btn" onclick="toggleConsole()">&#25511;&#21046;&#21488;</button><button class="icon-btn" onclick="fs()">&#20840;&#23631;</button></div></header>
<section class="console" id="console"><div class="console-top"><div class="address"><span class="address-label">PUBLIC LINK</span><a id="publicLink" href="#" target="_blank">CONNECTING...</a><span class="copy-state" id="copyState"></span><button class="icon-btn" onclick="copyAddress()">COPY</button></div><div class="profile" id="profile"><button data-p="low" onclick="setProfile('low')">&#20302;&#24310;&#36831;</button><button data-p="balanced" onclick="setProfile('balanced')">&#22343;&#34913;</button><button data-p="hd" onclick="setProfile('hd')">&#39640;&#28165;</button></div><button class="icon-btn collapse" onclick="toggleConsole()" id="collapseBtn">&#25910;&#36215;</button></div>
<div class="controls"><button class="ctrl" id="modeBtn" onclick="toggleMode()"><span class="ico">AI</span><span>&#20154;&#29289;&#35782;&#21035;</span></button><button class="ctrl" id="poseBtn" onclick="togglePose()"><span class="ico">P</span><span>&#23039;&#24577;&#35782;&#21035;</span></button><button class="ctrl" id="camBtn" onclick="toggleMulti()"><span class="ico">GRID</span><span>&#22810;&#25668;&#21516;&#23631;</span></button><button class="ctrl" id="pickBtn" onclick="openDrawer('camDrawer')"><span class="ico">CAM</span><span>&#20999;&#25442;&#25668;&#20687;&#22836;</span></button><button class="ctrl" id="autoBtn" onclick="toggleAuto()"><span class="ico">AUTO</span><span>&#33258;&#21160;&#25293;&#29031;</span></button><button class="ctrl" id="snapBtn" onclick="snap()"><span class="ico">SHOT</span><span>&#31435;&#21363;&#25293;&#29031;</span></button><button class="ctrl" id="recBtn" onclick="rec()"><span class="ico">REC</span><span>&#24405;&#20687;</span></button><button class="ctrl" onclick="openPlayback()"><span class="ico">PLAY</span><span>&#22238;&#25918;</span></button><button class="ctrl" onclick="fs()"><span class="ico">FS</span><span>&#20840;&#23631;</span></button></div>
<div class="audio-row"><button class="ctrl sound" id="soundBtn" onclick="toggleSound()"><span>&#24320;&#21551;&#22768;&#38899;</span></button><select id="audioSelect" onchange="changeAudio()"><option>&#27491;&#22312;&#35835;&#21462;&#40614;&#20811;&#39118;...</option></select><span class="stream-note" id="streamNote">--</span></div></section>
<div class="drawer" id="camDrawer"><div class="drawer-card"><div class="drawer-head"><h2>&#25668;&#20687;&#22836;&#30697;&#38453;</h2><button class="drawer-close" onclick="closeDrawer('camDrawer')">&#20851;&#38381;</button></div><div class="cam-grid" id="camlist"></div></div></div>
<div class="drawer" id="playDrawer"><div class="drawer-card"><div class="drawer-head"><h2>&#24405;&#20687;&#19982;&#25235;&#25293;&#22238;&#25918;</h2><button class="drawer-close" onclick="closeDrawer('playDrawer')">&#20851;&#38381;</button></div><div class="file-grid" id="filelist"></div></div></div>
<script>
const TXT={expand:"\u5c55\u5f00",collapse:"\u6536\u8d77",noCam:"\u672a\u53d1\u73b0\u53ef\u7528\u6444\u50cf\u5934",camera:"\u6444\u50cf\u5934",noMic:"\u6ca1\u6709\u53ef\u7528\u9ea6\u514b\u98ce",copied:"\u5df2\u590d\u5236",detect:"\u4eba\u7269",detectOn:"\u4eba\u7269\u8bc6\u522b\u5df2\u5f00",detectOpen:"\u5f00\u542f\u4eba\u7269\u8bc6\u522b",pose:"\u59ff\u6001",poseOn:"\u59ff\u6001\u8bc6\u522b\u5df2\u5f00",loading:"\u6a21\u578b\u52a0\u8f7d\u4e2d",multi:"\u591a\u6444\u540c\u5c4f",single:"\u5207\u56de\u5355\u6444",switching:"\u5207\u6362\u4e2d...",auto:"\u81ea\u52a8\u62cd\u7167",autoOn:"\u81ea\u52a8\u62cd\u7167\u5df2\u5f00",record:"\u5f55\u50cf",recordStop:"\u505c\u6b62\u5f55\u50cf",soundOn:"\u5f00\u542f\u58f0\u97f3",soundOff:"\u5173\u95ed\u58f0\u97f3",saved:"\u5df2\u4fdd\u5b58 \u2713",snap:"\u7acb\u5373\u62cd\u7167",reading:"\u6b63\u5728\u8bfb\u53d6\u56de\u653e...",empty:"\u6682\u65e0\u5f55\u50cf\u6216\u7167\u7247"};
const img=document.getElementById('v');let layout='tile',soundOn=false,audioAbort=null,audioCtx=null,nextAudioTime=0,audioRate=16000;let camIndices=[],camLabels=[],curCam=null,lastAddress='',consoleCompact=false;
function wantedLayout(){return innerWidth/Math.max(innerHeight,1)>1.08?'tile':'stack'}function applyLayout(force=false){const next=wantedLayout();if(force||next!==layout){layout=next;img.src='/stream.mjpg?layout='+layout+'&t='+Date.now()}}function fs(){const e=document.documentElement;if(document.fullscreenElement){document.exitFullscreen();return}document.body.classList.add('fullscreen-view');Promise.resolve((e.requestFullscreen||e.webkitRequestFullscreen).call(e)).catch(()=>document.body.classList.remove('fullscreen-view'))}document.addEventListener('fullscreenchange',()=>{if(!document.fullscreenElement)document.body.classList.remove('fullscreen-view')});function toggleConsole(){consoleCompact=!consoleCompact;document.getElementById('console').classList.toggle('compact',consoleCompact);document.getElementById('collapseBtn').textContent=consoleCompact?TXT.expand:TXT.collapse}function openDrawer(id){document.getElementById(id).style.display='block';if(id==='camDrawer')renderCamList()}function closeDrawer(id){document.getElementById(id).style.display='none'}
function renderCamList(){const list=document.getElementById('camlist');list.innerHTML='';if(!camIndices.length){list.innerHTML='<div class="empty">'+TXT.noCam+'</div>';return}camIndices.forEach((idx,i)=>{const b=document.createElement('button');b.className='cam-item'+(idx===curCam?' on':'');b.innerHTML='<b>CAM '+String(idx).padStart(2,'0')+'</b><br><small>'+(camLabels[i]||(TXT.camera+idx))+'</small>';b.onclick=()=>fetch('/cmd?action=select_cam&idx='+idx).then(()=>{curCam=idx;closeDrawer('camDrawer');setTimeout(refresh,500)});list.appendChild(b)})}
async function loadAudioDevices(){try{const d=await fetch('/audio/devices',{cache:'no-store'}).then(r=>r.json()),x=document.getElementById('audioSelect');x.innerHTML='';(d.devices||[]).forEach(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label||v.name||('MIC '+v.id);if(String(v.id)===String(d.selected))o.selected=true;x.appendChild(o)});if(!x.options.length)x.innerHTML='<option>'+TXT.noMic+'</option>'}catch(e){}}
function schedulePCM(bytes){if(!audioCtx||!bytes.length)return;const n=bytes.length>>1,buf=audioCtx.createBuffer(1,n,audioRate),out=buf.getChannelData(0),dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);for(let i=0;i<n;i++)out[i]=dv.getInt16(i*2,true)/32768;const src=audioCtx.createBufferSource();src.buffer=buf;src.connect(audioCtx.destination);const now=audioCtx.currentTime;if(nextAudioTime<now+.035)nextAudioTime=now+.035;src.start(nextAudioTime);nextAudioTime+=buf.duration;if(nextAudioTime-now>.45)nextAudioTime=now+.06}
async function startSound(){try{audioAbort=new AbortController();audioCtx=audioCtx||new(window.AudioContext||window.webkitAudioContext)();await audioCtx.resume();soundOn=true;const b=document.getElementById('soundBtn');b.querySelector('span').textContent=TXT.soundOff;b.classList.add('on');nextAudioTime=audioCtx.currentTime+.08;let seq=0;while(soundOn){const r=await fetch('/audio/chunk?after='+seq+'&t='+Date.now(),{signal:audioAbort.signal,cache:'no-store'});if(!r.ok)throw new Error(await r.text());audioRate=parseInt(r.headers.get('X-Sample-Rate')||'16000');seq=parseInt(r.headers.get('X-Audio-Seq')||seq);const pcm=new Uint8Array(await r.arrayBuffer());if(soundOn&&pcm.length)schedulePCM(pcm)}}catch(e){if(soundOn)console.log(e)}finally{if(soundOn)stopSound()}}function stopSound(){soundOn=false;if(audioAbort)audioAbort.abort();audioAbort=null;const b=document.getElementById('soundBtn');b.querySelector('span').textContent=TXT.soundOn;b.classList.remove('on')}function toggleSound(){soundOn?stopSound():startSound()}async function changeAudio(){const id=document.getElementById('audioSelect').value,was=soundOn;stopSound();await fetch('/audio/select?device='+encodeURIComponent(id));if(was)setTimeout(startSound,220)}
function compactUrl(u){try{const x=new URL(u),h=x.host;return h.length<34?h:h.slice(0,13)+'...'+h.slice(-17)}catch(e){return u}}async function copyAddress(){if(!lastAddress)return;try{await navigator.clipboard.writeText(lastAddress);document.getElementById('copyState').textContent=TXT.copied;setTimeout(()=>document.getElementById('copyState').textContent='',1200)}catch(e){}}function setProfile(p){fetch('/cmd?action=stream_profile&idx='+p).then(()=>setTimeout(refresh,350))}function setButton(id,on,text){const b=document.getElementById(id);if(!b)return;b.classList.toggle('on',!!on);if(text)b.lastElementChild.textContent=text}
async function refresh(){const t=performance.now();try{const s=await fetch('/status?t='+Date.now(),{cache:'no-store'}).then(r=>r.json()),rtt=Math.round(performance.now()-t);camIndices=s.camera_indices||[];camLabels=s.camera_labels||[];curCam=s.current_camera;lastAddress=s.short_url||s.remote||s.lan||'';const link=document.getElementById('publicLink');link.href=lastAddress||'#';link.textContent=s.short_url||compactUrl(lastAddress)||'CONNECTING...';link.title=s.remote||lastAddress;const persons=Number(s.detected_people)||0;document.getElementById('modeMetric').textContent=s.pose_enabled?(TXT.pose+' '+persons):(s.detection_enabled===false?'MONITOR':(TXT.detect+' '+persons));document.getElementById('fpsMetric').textContent=(Number(s.stream_fps)||0).toFixed(1)+' FPS';document.getElementById('camMetric').textContent=(s.active_camera_count||1)+' / '+(s.camera_count||1);document.getElementById('netMetric').textContent=rtt+' ms';document.getElementById('liveText').textContent=s.remote?'PUBLIC LIVE':'LAN LIVE';document.getElementById('streamNote').textContent=(s.stream_width||'--')+'px / Q'+(s.stream_quality||'--')+' / '+(s.jpeg_kb||0).toFixed(0)+'KB';setButton('modeBtn',s.detection_enabled!==false,s.detection_enabled===false?TXT.detectOpen:TXT.detectOn);const pb=document.getElementById('poseBtn');pb.disabled=!!s.pose_loading;setButton('poseBtn',s.pose_enabled,s.pose_loading?TXT.loading:(s.pose_enabled?TXT.poseOn:TXT.pose));setButton('camBtn',s.multi_camera,s.camera_switching?TXT.switching:(s.multi_camera?TXT.single:TXT.multi));document.getElementById('pickBtn').style.display=s.multi_camera?'none':'';setButton('autoBtn',s.auto_snap,s.auto_snap?TXT.autoOn:TXT.auto);setButton('recBtn',s.recording,s.recording?TXT.recStop:TXT.record);document.getElementById('recBtn').classList.toggle('rec',!!s.recording);document.querySelectorAll('#profile button').forEach(b=>b.classList.toggle('on',b.dataset.p===(s.stream_profile||'balanced')))}catch(e){document.getElementById('liveText').textContent='RECONNECTING'}}
function toggleMode(){fetch('/cmd?action=detection_toggle').then(()=>setTimeout(refresh,250))}function togglePose(){document.getElementById('poseBtn').lastElementChild.textContent=TXT.switching;fetch('/cmd?action=pose_toggle').then(()=>{setTimeout(refresh,300);setTimeout(refresh,1500)})}function toggleMulti(){document.getElementById('camBtn').lastElementChild.textContent=TXT.switching;fetch('/cmd?action=multi_toggle').then(()=>setTimeout(refresh,1800))}function toggleAuto(){fetch('/cmd?action=auto_snap_toggle').then(()=>setTimeout(refresh,220))}function snap(){fetch('/cmd?action=snapshot').then(()=>{const b=document.getElementById('snapBtn');b.lastElementChild.textContent=TXT.saved;setTimeout(()=>b.lastElementChild.textContent=TXT.snap,900)})}function rec(){fetch('/cmd?action=record').then(()=>setTimeout(refresh,260))}
function openPlayback(){openDrawer('playDrawer');const box=document.getElementById('filelist');box.innerHTML='<div class="empty">'+TXT.reading+'</div>';fetch('/files',{cache:'no-store'}).then(r=>r.json()).then(items=>{box.innerHTML='';if(!items.length){box.innerHTML='<div class="empty">'+TXT.empty+'</div>';return}items.forEach(it=>{const d=document.createElement('div');d.className='file-item';const media=it.type==='video'?'<video src="'+it.url+'" controls preload="metadata"></video>':'<img src="'+it.url+'" loading="lazy">';d.innerHTML=media+'<div class="file-cap">'+it.name+'</div>';box.appendChild(d)})})}
window.addEventListener('resize',()=>applyLayout());window.addEventListener('orientationchange',()=>setTimeout(()=>applyLayout(true),250));document.addEventListener('visibilitychange',()=>{if(!document.hidden)applyLayout(true)});applyLayout(true);loadAudioDevices();refresh();setInterval(refresh,1200);
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


STREAM_PROFILES = {
    # name: (max_width, jpeg_quality, fps_cap)
    # Quick tunnels are bandwidth-limited, so the balanced preset avoids the
    # 120+ KB frames that otherwise collapse public playback to 3-5 FPS.
    "low": (768, 68, 24),
    "balanced": (1024, 76, 20),
    "hd": (1280, 82, 14),
}


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
        self.profile_name = "balanced"
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

    def set_profile(self, profile):
        """Switch the shared encoder profile without restarting the camera/server."""
        cfg = STREAM_PROFILES.get(str(profile).lower())
        if cfg is None:
            return False
        width, quality, fps = cfg
        self.max_width = int(width)
        self.quality = int(quality)
        self.fps_cap = max(int(fps), 1)
        self.min_interval = 1.0 / self.fps_cap
        self.submit_interval = self.min_interval * 0.82
        self.profile_name = str(profile).lower()
        # Let the next camera frame enter immediately after switching profile.
        self._last_submit = 0.0
        return True

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

    # ---------- 瀹夊叏鍙栨枃浠?----------
    def _safe_path(self, root, name):
        if root is None or not name:
            return None
        # 娴忚鍣ㄤ細鎶婃枃浠跺悕涓殑涓枃/绌烘牸鍋氱櫨鍒嗗彿缂栫爜锛堝 鍏ㄥ睆 -> %E5%85%A8%E5%B1%8F锛夛紝
        # 蹇呴』鍏堣В鐮佹墠鑳藉尮閰嶇鐩樹笂鐨勭湡瀹炴枃浠跺悕锛屽惁鍒欎腑鏂囧悕蹇収浼?404 鏃犳硶鏄剧ず銆?
        name = unquote(name)
        base = os.path.basename(name)  # 闃茬洰褰曠┛瓒?
        p = os.path.abspath(os.path.join(root, base))
        if os.path.dirname(p) != os.path.abspath(root):
            return None
        return p if os.path.isfile(p) else None

    # ---------- 鏈嶅姟鍚姩/鍋滄 ----------
    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            wbufsize = 0

            def setup(self):
                super().setup()
                try:
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass  # 闈欓粯鏃ュ織

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
                # 鎵嬫満绔彲甯??layout=tile|stack 鍒囨崲澶氭憚甯冨眬锛堟í灞忓钩閾?绔栧睆绔栨帓锛?
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
                idx = q.get("idx", [""])[0]
                try:
                    if streamer.on_command is not None:
                        msg = streamer.on_command(action, idx)
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
            print(f"[鎺ㄦ祦] 绔彛 {self.port} 鍚姩澶辫触: {e}")
            return False
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print("\n" + "=" * 46)
        print("  Remote mobile monitoring started")
        print(f"  LAN: http://{self.lan_ip}:{self.port}")
        print("  Public URL will appear after tunnel connection")
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
    """Build an OpenCV BGR QR image, or return None when qrcode is unavailable."""
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
