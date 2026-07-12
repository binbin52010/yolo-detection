"""
remote_access.py
----------------
免同一 WiFi 的远程观看：把本机 MJPEG 服务暴露到公网，手机用流量/异地网络
打开公网 https 地址即可实时观看监控画面。

中继策略（按可靠性排序，自动回退）：
  1. Cloudflare 快速隧道（cloudflared quick tunnel）—— 首选。
     - 零账号、零注册，直接返回一个 https://*.trycloudflare.com 公网地址。
     - 手机浏览器（含流量/异地）可直接打开，无需客户端。
     - 二进制随项目打包在 bin/cloudflared.exe；缺失时自动从官网下载。
  2. SSH 反向隧道（serveo.net / localhost.run）—— 兜底。
     - 匿名、免费，复用 Git 自带 ssh。
     - 自动把 ssh 的 HOME 指向纯 ASCII 目录，绕开中文用户名导致
       “Could not create directory ~/.ssh” 的失败。
     - 用 bash -c 包装 + 二进制读取线程，避免 Windows 管道缓冲卡死。

隧道断开自动重连；所有中继连续失败过多后停止尝试并打印替代方案。
"""

import os
import re
import sys
import time
import shlex
import ssl
import shutil
import subprocess
import threading
import urllib.request


# ============ Cloudflared（首选） ============
_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
CLOUDFLARED_EXE = os.path.join(_BIN_DIR, "cloudflared.exe")
CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-windows-amd64.exe")
_CF_URL_RE = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")

# ============ SSH 中继（兜底） ============
SSH_RELAYS_DEF = [
    {"host": "serveo.net", "remote": "80",
     "tunnel_re": re.compile(r"https?://[A-Za-z0-9\-]+\.serveo\.net")},
    {"host": "localhost.run", "remote": "80",
     "tunnel_re": re.compile(r"https?://[A-Za-z0-9\-]+\.lhr\.life")},
]


_url_re = re.compile(r"https?://[^\s'\"<>]+")

# Windows 下，当父进程（pythonw）没有控制台时，启动控制台子系统程序
# （cloudflared.exe / ssh.exe）会单独弹出一个黑色控制台窗口。用
# CREATE_NO_WINDOW 让子进程静默运行，既不弹黑窗口，也避免用户误关黑窗口
# 把隧道进程一起杀掉导致手机连不上。
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _tunnel_home():
    """纯 ASCII 目录作为 ssh 的 HOME，避免中文路径（用户名或项目目录）被
    MSYS/Git 的 ssh 错误编码导致卡死。优先用 C:\\Users\\Public（全 ASCII
    且通常可写），失败回退到脚本目录。"""
    d = r"C:\Users\Public\.yolo_tunnel_home"
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, ".w"), "w") as f:
            f.write("1")
        os.remove(os.path.join(d, ".w"))
        return d
    except Exception:
        pass
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tunnel_home")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _find_ssh():
    # 优先 PATH 中的 ssh（Git Bash 下即 Git 自带的 ssh，实测可建立隧道）；
    # 其次回退常见绝对路径。注意：Windows 系统 OpenSSH 在 Python 子进程下
    # 连接 serveo 会读取卡死（无输出），故不作为首选。
    p = shutil.which("ssh")
    if p:
        return p
    candidates = [
        r"C:\Program Files\Git\usr\bin\ssh.exe",
        r"C:\Program Files (x86)\Git\usr\bin\ssh.exe",
        r"C:\Windows\System32\OpenSSH\ssh.exe",
        r"C:\Windows\System32\ssh.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _find_bash():
    return shutil.which("bash")


def _ensure_cloudflared(verbose=True):
    """确保 cloudflared 二进制存在；缺失则从官网下载。返回路径或 None。"""
    if os.path.isfile(CLOUDFLARED_EXE):
        return CLOUDFLARED_EXE
    if verbose:
        print("[公网] 未发现 cloudflared，尝试从官网下载（约 50MB）...")
    try:
        os.makedirs(_BIN_DIR, exist_ok=True)
        req = urllib.request.Request(
            CLOUDFLARED_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(
                req, timeout=180, context=ssl.create_default_context()) as r:
            data = r.read()
        with open(CLOUDFLARED_EXE, "wb") as f:
            f.write(data)
        if verbose:
            print(f"[公网] cloudflared 已就绪（{len(data)//1024//1024} MB）")
        return CLOUDFLARED_EXE
    except Exception as e:
        if verbose:
            print(f"[公网] cloudflared 下载失败: {e}")
        return None


# ============ 中继抽象 ============
class Relay:
    name = "?"

    def launch(self, local_port, env):
        """启动隧道进程，返回 Popen 对象；无法启动返回 None。"""
        raise NotImplementedError

    def parse(self, line, on_url):
        """解析一行输出，若包含本中继的隧道地址则调用 on_url 并返回 True。"""
        m = _url_re.search(line)
        if m:
            url = m.group(0).rstrip(".,);")
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]
            on_url(url)
            return True
        return False

    def available(self):
        return True


class CloudflaredRelay(Relay):
    name = "cloudflared"

    def available(self):
        return _ensure_cloudflared(verbose=False) is not None

    def launch(self, local_port, env):
        exe = _ensure_cloudflared()
        if not exe:
            return None
        cmd = [exe, "tunnel", "--url",
               f"http://localhost:{local_port}", "--no-autoupdate"]
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=0, env=env,
                                    creationflags=_NO_WINDOW)
        except Exception as e:
            print(f"[公网] 启动 cloudflared 失败: {e}")
            return None

    def parse(self, line, on_url):
        m = _CF_URL_RE.search(line)
        if m:
            on_url(m.group(0))
            return True
        return False


class SshRelay(Relay):
    def __init__(self, host, remote, tunnel_re, ssh, home):
        self.name = host
        self.host = host
        self.remote = remote
        self.tunnel_re = tunnel_re
        self.ssh = ssh
        self.home = home

    def available(self):
        return self.ssh is not None

    def launch(self, local_port, env):
        kh = os.path.join(self.home, "known_hosts")
        ssh_args = [
            self.ssh,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={kh}",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-nNT",
            "-R", f"{self.remote}:localhost:{local_port}",
            self.host,
        ]
        bash = _find_bash()
        if bash:
            quoted = " ".join(shlex.quote(a) for a in ssh_args)
            cmd = [bash, "-c", quoted]
        else:
            cmd = ssh_args
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=0, env=env,
                                    creationflags=_NO_WINDOW)
        except Exception as e:
            print(f"[公网] 启动 ssh({self.host}) 失败: {e}")
            return None

    def parse(self, line, on_url):
        # 仅接受本中继真正的隧道地址（避免把帮助页/文档 URL 误判为隧道地址）
        m = self.tunnel_re.search(line)
        if m:
            url = m.group(0)
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]
            on_url(url)
            return True
        return False


class Tunnel:
    def __init__(self, local_port, on_url=None):
        self.local_port = local_port
        self.on_url = on_url
        self.ssh = _find_ssh()
        self.home = _tunnel_home()
        self.running = True
        self._thread = None
        self._proc = None
        self._lock = threading.Lock()
        self.current_url = None
        self.last_error = None
        self._got_url = False
        self.relay_index = 0
        self.fail_streak = 0
        # 中继列表：cloudflared 优先，SSH 中继兜底
        self.relays = [CloudflaredRelay()]
        if self.ssh is not None:
            for r in SSH_RELAYS_DEF:
                self.relays.append(
                    SshRelay(r["host"], r["remote"], r["tunnel_re"],
                             self.ssh, self.home))

    def _env(self):
        # 仅覆盖 HOME（供 Git-bash 的 ssh 使用 ASCII 目录）；
        # 系统 OpenSSH 使用原 USERPROFILE（可正确处理中文用户名），不覆盖它。
        env = dict(os.environ)
        env["HOME"] = self.home
        return env

    def start(self):
        if not self.relays:
            print("[公网] 没有可用的公网中继（缺少 cloudflared 且未找到 ssh）。\n"
                  "        请用 ngrok / frp 等把本机 "
                  f"{self.local_port} 端口暴露到公网，再把地址发给手机。")
            return False
        print(f"[公网] 中继顺序(按可靠性): "
              f"{', '.join(r.name for r in self.relays)}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _emit(self, url):
        with self._lock:
            self.current_url = url
            self._got_url = True
        if self.on_url:
            try:
                self.on_url(url)
            except Exception:
                pass

    def _run(self):
        env = self._env()
        while self.running:
            if self.fail_streak > len(self.relays) * 3:
                print("[公网] 多次尝试均失败，停止公网隧道。\n"
                      "        可能原因：本机无法访问外网 / 中继被屏蔽。\n"
                      "        替代方案：ngrok / cloudflared / frp 暴露本机 "
                      f"{self.local_port} 端口。")
                break

            relay = self.relays[self.relay_index % len(self.relays)]
            self.relay_index += 1

            try:
                proc = relay.launch(self.local_port, env)
            except Exception as e:
                proc = None
                print(f"[公网] 中继 {relay.name} 启动异常: {e}")

            if proc is None:
                self.fail_streak += 1
                time.sleep(2)
                continue

            with self._lock:
                self._proc = proc
            self._got_url = False
            print(f"[公网] 正在通过 {relay.name} 建立隧道"
                  f"（本地端口 {self.local_port}）...")

            def reader():
                buf = b""
                try:
                    while True:
                        chunk = proc.stdout.read(512)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line_b, buf = buf.split(b"\n", 1)
                            line = line_b.decode(errors="replace").rstrip("\r")
                            if relay.parse(line, self._emit):
                                pass
                except Exception:
                    pass
                if buf:
                    relay.parse(buf.decode(errors="replace").rstrip("\r"),
                                self._emit)

            rt = threading.Thread(target=reader, daemon=True)
            rt.start()

            _start = time.time()
            while self.running:
                if proc.poll() is not None:
                    break
                if time.time() - _start > 30 and not self._got_url:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
                time.sleep(0.2)

            if self._got_url:
                self.fail_streak = 0
                # 隧道保持中，等待其断开后自动重连
                while self.running and proc.poll() is None:
                    time.sleep(0.5)
            else:
                self.fail_streak += 1
                self.last_error = f"relay {relay.name} 未返回地址"
                try:
                    proc.kill()
                except Exception:
                    pass
                rt.join(timeout=2)
                if self.running:
                    print("[公网] 本次未建立成功，稍后尝试其它节点...")
                    time.sleep(3)

        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass


def start_tunnel(local_port, on_url=None):
    """启动公网隧道，返回 Tunnel 控制器；无任何可用中继返回 None。"""
    t = Tunnel(local_port, on_url)
    if t.start():
        return t
    return None


if __name__ == "__main__":
    # 自检：验证 cloudflared / ssh 可用性、正则匹配、命令拼接
    t = Tunnel(8000)
    print("中继列表:", [r.name for r in t.relays])
    print("cloudflared 路径:", CLOUDFLARED_EXE,
          "存在:", os.path.isfile(CLOUDFLARED_EXE))
    print("ssh:", t.ssh)
    print("home:", t.home)
    print("CF 正则示例:",
          _CF_URL_RE.findall("Your quick Tunnel has been created! "
                             "https://abc-def.trycloudflare.com"))
    if t.ssh:
        print("serveo tunnel_re 示例:",
              SSH_RELAYS_DEF[0]["tunnel_re"].findall(
                  "Forwarding HTTP traffic from https://x.serveo.net"))
