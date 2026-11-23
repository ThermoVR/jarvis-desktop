#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.11.*"
# dependencies = ["numpy", "sgp4"]
# ///
"""Hey Claude live face wallpaper.

A neon 3D wireframe hologram head drawn full screen behind your windows. It shows what
the voice assistant is doing:

  idle      no voice session: dim face, slow blinks
  thinking  a wake-word session is running but not talking: eyes dart, ring spins
  listening VoiceMode holds the mic: eyes wide, cyan pulses
  speaking  VoiceMode is talking: mouth is a waveform driven by speaker volume

State sources (no changes to other programs needed):
  - `claude -p` started by hey_claude.py  -> session running
  - flock on ~/.voicemode/conch            -> a converse call is speaking/listening
  - level of the default speaker monitor   -> speaking vs listening, mouth size

Run:  ./face_wallpaper.py          (Ctrl+C or `pkill -f face_wallpaper` to stop)
Env:  FACE_LEFT_MARGIN (px kept free on the left for desktop icons, default 0),
      FACE_FPS (default 30), FACE_DEMO=1 cycles through the states for testing
      FACE_WEATHER=rain|snow|storm[:0..1] forces the weather effects (testing);
      live weather location: ~/.config/hey-claude/weather.conf ("LAT LON")
"""
import datetime
import fcntl
import json
import math
import urllib.request
import os
import random
import subprocess
import threading
import time
import tkinter as tk

import numpy as np
import sky
from array import array
from pathlib import Path

FPS = int(os.environ.get("FACE_FPS", "20"))
LEFT_MARGIN = int(os.environ.get("FACE_LEFT_MARGIN", "0"))  # px left free for desktop icons
DEMO = os.environ.get("FACE_DEMO") == "1"
CONCH = Path.home() / ".voicemode" / "conch"
LOG_PERF = Path(__file__).resolve().parent / "logs" / "perf.log"
EVENTS_DIR = Path.home() / ".voicemode" / "logs" / "events"
FORCE_FEM = os.environ.get("FACE_FEM")  # "1"/"0" forces the female/male head (testing)
SESSION_PATTERN = "claude -p The user just woke you"
# Live weather (Open-Meteo, no key). The location is "LAT LON" in this file.
WEATHER_CONF = Path.home() / ".config" / "hey-claude" / "weather.conf"
FORCE_WEATHER = os.environ.get("FACE_WEATHER")  # testing: "rain", "snow:0.5", "storm", "clear"
N_RAIN, N_SPLASH, N_SNOW = 260, 48, 220
BG = (3, 1, 6)

STATE_COLORS = {  # (primary, secondary, brightness)
    "idle": ((210, 35, 50), (180, 25, 45), 0.75),
    "thinking": ((255, 70, 40), (255, 20, 40), 0.9),
    "listening": ((225, 235, 255), (255, 25, 50), 1.0),
    "speaking": ((255, 35, 55), (235, 240, 255), 1.0),
}
LABELS = {
    "idle": "STANDBY",
    "thinking": "PROCESSING",
    "listening": "LISTENING",
    "speaking": "TRANSMITTING",
}


def hexcol(rgb, k=1.0):
    """Colour scaled towards the background by k (0 = background, 1 = full)."""
    r, g, b = (int(BG[i] + (rgb[i] - BG[i]) * max(0.0, min(1.0, k))) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_rgb(a, b, t):
    return tuple(lerp(a[i], b[i], t) for i in range(3))


# ---- state detection -------------------------------------------------------------
class Sensors:
    def __init__(self):
        self.level = 0.0          # smoothed speaker level 0..1
        self.last_loud = 0.0
        self.conch = False
        self.session = False
        self.voice = ""           # voice of the most recent TTS line (from VoiceMode events)
        self.phase = None         # latest audio-phase event type (see PHASE_EVENTS)
        self._stop = False
        threading.Thread(target=self._audio_loop, daemon=True).start()
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _audio_loop(self):
        while not self._stop:
            try:
                p = subprocess.Popen(
                    ["parec", "-d", "@DEFAULT_MONITOR@", "--format=s16le", "--rate=8000",
                     "--channels=1", "--latency-msec=40", "--client-name=face-wallpaper"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                while True:
                    data = p.stdout.read(512)  # 256 samples, 32 ms
                    if not data:
                        break
                    s = array("h", data)
                    rms = math.sqrt(sum(x * x for x in s) / max(1, len(s))) / 32768
                    v = min(1.0, rms * 6)
                    self.level = max(v, self.level * 0.8)
                    if v > 0.03:
                        self.last_loud = time.monotonic()
            except Exception:
                pass
            time.sleep(2)

    def _poll_loop(self):
        while not self._stop:
            self.conch = conch_held()
            try:
                self.session = subprocess.run(
                    ["pgrep", "-f", SESSION_PATTERN], capture_output=True).returncode == 0
            except Exception:
                self.session = False
            self.voice = last_tts_voice() or self.voice
            self.phase = last_phase_event()
            time.sleep(0.25)

    def state(self):
        if DEMO:
            return ["idle", "thinking", "listening", "speaking"][int(time.monotonic() / 5) % 4]
        if self.conch:
            if time.monotonic() - self.last_loud < 0.6:
                return "speaking"
            # only claim "listening" while the mic is really recording; while Kokoro is still
            # generating the first audio, or Whisper is transcribing, show "thinking"
            if self.phase == "RECORDING_START":
                return "listening"
            if self.phase == "TTS_FIRST_AUDIO":
                return "speaking"
            return "thinking"
        if self.session:
            return "thinking"
        return "idle"

    def feminine(self):
        """True when the current voice is a female one (Kokoro ids: 2nd letter 'f', e.g. ff_siwis)."""
        if FORCE_FEM is not None:
            return FORCE_FEM == "1"
        if DEMO:
            return int(time.monotonic() / 20) % 2 == 1
        return self.voice[1:2] == "f"

    def mouth_level(self):
        if DEMO:
            t = time.monotonic()
            return abs(math.sin(t * 9) * math.sin(t * 3.1)) * 0.9
        return self.level


WMO = {  # Open-Meteo weather code -> (kind, intensity 0..1, HUD label)
    0: ("clear", 0, "CLEAR"), 1: ("clear", 0, "MOSTLY CLEAR"), 2: ("clear", 0, "PARTLY CLOUDY"),
    3: ("clear", 0, "OVERCAST"), 45: ("clear", 0, "FOG"), 48: ("clear", 0, "FREEZING FOG"),
    51: ("rain", .25, "LIGHT DRIZZLE"), 53: ("rain", .35, "DRIZZLE"), 55: ("rain", .5, "HEAVY DRIZZLE"),
    56: ("rain", .3, "FREEZING DRIZZLE"), 57: ("rain", .5, "FREEZING DRIZZLE"),
    61: ("rain", .4, "LIGHT RAIN"), 63: ("rain", .65, "RAIN"), 65: ("rain", 1, "HEAVY RAIN"),
    66: ("rain", .4, "FREEZING RAIN"), 67: ("rain", .9, "FREEZING RAIN"),
    71: ("snow", .35, "LIGHT SNOW"), 73: ("snow", .65, "SNOW"), 75: ("snow", 1, "HEAVY SNOW"),
    77: ("snow", .3, "SNOW GRAINS"), 80: ("rain", .45, "SHOWERS"), 81: ("rain", .7, "SHOWERS"),
    82: ("rain", 1, "VIOLENT SHOWERS"), 85: ("snow", .5, "SNOW SHOWERS"), 86: ("snow", 1, "SNOW SHOWERS"),
    95: ("storm", .8, "THUNDERSTORM"), 96: ("storm", .9, "THUNDERSTORM + HAIL"),
    99: ("storm", 1, "THUNDERSTORM + HAIL"),
}


class Weather:
    """Current weather for the location in WEATHER_CONF, refreshed every 10 minutes."""

    def __init__(self):
        self.kind, self.level, self.wind_x, self.temp, self.label = "clear", 0.0, 0.0, None, ""
        self.cloud, self.feels = None, None
        if FORCE_WEATHER:
            kind, _, lvl = FORCE_WEATHER.partition(":")
            self.kind, self.level = kind, float(lvl or 0.8) if kind != "clear" else 0.0
            self.wind_x, self.label = 0.4, f"{kind.upper()} (TEST)"
            return
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def location():
        try:
            nums = " ".join(l.split("#")[0] for l in WEATHER_CONF.read_text().splitlines()).replace(",", " ").split()
            return float(nums[0]), float(nums[1])
        except (OSError, IndexError, ValueError):
            return None

    def _loop(self):
        while True:
            delay = 600
            loc = self.location()
            if loc:
                try:
                    self._fetch(*loc)
                except Exception as e:
                    print("weather fetch failed:", e, flush=True)
                    delay = 120
            time.sleep(delay)

    def _fetch(self, lat, lon):
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat:.2f}&longitude={lon:.2f}"
               "&current=temperature_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
               "cloud_cover,apparent_temperature")
        with urllib.request.urlopen(url, timeout=15) as r:
            cur = json.load(r)["current"]
        code = int(cur["weather_code"])
        kind, lvl, label = WMO.get(code, ("clear", 0, f"CODE {code}"))
        if kind != "clear":
            lvl = max(lvl, min(1.0, (cur.get("precipitation") or 0) / 6))   # mm in the last 15 min
        # wind_direction is where the wind comes from: a west wind (270) pushes drops to the right
        wdir, speed = math.radians(cur["wind_direction_10m"]), cur["wind_speed_10m"]
        self.wind_x = -math.sin(wdir) * min(1.0, speed / 50)
        self.temp, self.kind, self.level, self.label = cur["temperature_2m"], kind, lvl, label
        self.cloud, self.feels = cur.get("cloud_cover"), cur.get("apparent_temperature")
        print(f"weather: {label} code={code} level={lvl:.2f} temp={self.temp} wind={speed}km/h", flush=True)

    def hud(self):
        if not self.label:
            return ""
        return f"  //  WX {self.label}" + (f" {self.temp:.0f}°C" if self.temp is not None else "")


def last_tts_voice():
    """Voice used by the latest TTS_START event in VoiceMode's newest event log."""
    try:
        f = max(EVENTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        with open(f, "rb") as fh:
            fh.seek(max(0, fh.seek(0, 2) - 16384))
            tail = fh.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    i = tail.rfind('"TTS_START"')
    if i < 0:
        return None
    j = tail.find('"voice": "', i)
    if j < 0 or tail.find("\n", i, j) >= 0:
        return None
    j += len('"voice": "')
    return tail[j:tail.find('"', j)]


PHASE_EVENTS = ("TTS_START", "TTS_FIRST_AUDIO", "TTS_PLAYBACK_END", "RECORDING_START",
                "RECORDING_END", "STT_START", "STT_COMPLETE", "SESSION_END")


def last_phase_event():
    """Type of the latest audio-phase event in VoiceMode's newest event log, or None."""
    try:
        f = max(EVENTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        with open(f, "rb") as fh:
            fh.seek(max(0, fh.seek(0, 2) - 8192))
            tail = fh.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    best, best_i = None, -1
    for ev in PHASE_EVENTS:
        i = tail.rfind(f'"event_type": "{ev}"')
        if i > best_i:
            best, best_i = ev, i
    return best


def conch_held() -> bool:
    try:
        fd = os.open(CONCH, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


# ---- keep the head out from under windows -----------------------------------------
class WindowDodger:
    """Polls the open windows every second and picks the screen x where the head is least covered."""

    def __init__(self, W, H, head_w, head_top, head_bot):
        self.W, self.H = W, H
        self.head_w, self.top, self.bot = head_w, head_top, head_bot
        self.target_x = W / 2
        self.focus = None          # (x, y) centre of the focused window, or None
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _xprop(*args):
        return subprocess.run(["xprop", *args], capture_output=True, text=True, timeout=2).stdout

    def _windows(self):
        root = self._xprop("-root", "_NET_CLIENT_LIST", "_NET_CURRENT_DESKTOP")
        ids, desk = [], "0"
        for line in root.splitlines():
            if line.startswith("_NET_CLIENT_LIST(") and "#" in line:
                ids = [w.strip() for w in line.split("#", 1)[1].split(",") if w.strip()]
            elif line.startswith("_NET_CURRENT_DESKTOP"):
                desk = line.split("=")[-1].strip()
        rects = []
        for wid in ids:
            props = self._xprop("-id", wid, "_NET_WM_WINDOW_TYPE", "_NET_WM_STATE", "_NET_WM_DESKTOP")
            if "HIDDEN" in props or ("TYPE_NORMAL" not in props and "TYPE_DIALOG" not in props):
                continue
            d = [l.split("=")[-1].strip() for l in props.splitlines() if l.startswith("_NET_WM_DESKTOP")]
            if d and d[0] not in (desk, "4294967295"):
                continue
            g = subprocess.run(["xdotool", "getwindowgeometry", "--shell", str(int(wid, 16))],
                               capture_output=True, text=True, timeout=2).stdout
            v = dict(l.split("=", 1) for l in g.splitlines() if "=" in l)
            try:
                rects.append((int(v["X"]), int(v["Y"]), int(v["WIDTH"]), int(v["HEIGHT"])))
            except (KeyError, ValueError):
                pass
        return rects

    def _coverage(self, cx, rects):
        xs = np.linspace(cx - self.head_w / 2, cx + self.head_w / 2, 12)
        ys = np.linspace(self.top, self.bot, 10)
        X, Y = np.meshgrid(xs, ys)
        cov = np.zeros_like(X, dtype=bool)
        for x, y, w, h in rects:
            cov |= (X >= x) & (X < x + w) & (Y >= y) & (Y < y + h)
        return cov.mean()

    def _loop(self):
        while True:
            try:
                rects = self._windows()
                lo, hi = self.head_w / 2, self.W - self.head_w / 2
                cands = np.linspace(lo, hi, 25)
                score = [self._coverage(c, rects) + 0.08 * abs(c - self.W / 2) / self.W for c in cands]
                best = cands[int(np.argmin(score))]
                cur = self._coverage(self.target_x, rects) + 0.08 * abs(self.target_x - self.W / 2) / self.W
                if not rects or self._coverage(self.W / 2, rects) <= 0.02:
                    self.target_x = self.W / 2      # centre is clear: go home
                elif min(score) < cur - 0.08:     # hysteresis: only move for a clear win
                    self.target_x = float(best)
            except Exception as e:
                print("dodger:", e, flush=True)
            try:
                self.focus = self._active_centre()
            except Exception:
                self.focus = None
            time.sleep(1.0)

    def _active_centre(self):
        wid = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True,
                             timeout=2).stdout.strip()
        if not wid:
            return None
        props = self._xprop("-id", wid, "_NET_WM_WINDOW_TYPE", "_NET_WM_STATE")
        if "HIDDEN" in props or ("TYPE_NORMAL" not in props and "TYPE_DIALOG" not in props):
            return None
        g = subprocess.run(["xdotool", "getwindowgeometry", "--shell", wid],
                           capture_output=True, text=True, timeout=2).stdout
        v = dict(l.split("=", 1) for l in g.splitlines() if "=" in l)
        return (int(v["X"]) + int(v["WIDTH"]) / 2, int(v["Y"]) + int(v["HEIGHT"]) / 2)


# ---- drawing ---------------------------------------------------------------------
class Showcase:
    """FACE_SHOWCASE=1: loop through weather and sky scenes to show the wallpaper off.
    Drives Weather, the sky clock, the moon phase, aurora, meteors and a fake ISS pass live."""

    @staticmethod
    def _ts(s):
        return datetime.datetime.fromisoformat(s).timestamp()

    def __init__(self, face):
        self.f = face
        T = self._ts
        # (seconds, HUD label, weather kind, level, cloud %, sky time start, sky time end, extras)
        self.scenes = [
            (26, "DAY TIME-LAPSE", "clear", 0, 0, T("2026-11-17T07:00"), T("2026-11-17T17:30"), {}),
            (15, "RAIN", "rain", 0.85, 70, T("2026-11-17T16:00"), T("2026-11-17T16:20"), {}),
            (16, "THUNDERSTORM", "storm", 1.0, 90, T("2026-11-17T17:30"), T("2026-11-17T17:50"), {}),
            (32, "AURORA + METEORS + ISS", "clear", 0, 0, T("2026-11-16T22:30"), T("2026-11-16T22:50"),
             {"kp": 8, "meteors": 30, "iss": 24}),
            (22, "MOON PHASES", "clear", 0, 0, T("2026-11-23T23:00"), T("2026-11-23T23:00"), {"phase": True}),
            (18, "SNOW UNDER A FULL MOON", "snow", 0.85, 15, T("2026-11-23T23:00"), T("2026-11-23T23:15"), {}),
            (26, "NIGHT TIME-LAPSE", "clear", 0, 0, T("2026-11-23T23:15"), T("2026-11-24T07:00"), {}),
        ]
        sky.SLOW_DT = 0.05
        sky.COLOUR_DT = 0.1
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        wx, space = self.f.weather, self.f.sky.space
        total = sum(s[0] for s in self.scenes)
        t0 = time.monotonic()
        current = None
        while True:
            t = (time.monotonic() - t0) % total
            for i, (dur, label, kind, lvl, cloud, a, b, extra) in enumerate(self.scenes):
                if t < dur:
                    break
                t -= dur
            f = t / dur
            if current != i:
                current = i
                print(f"showcase: {label}", flush=True)
                wx.kind, wx.level, wx.label, wx.temp = kind, lvl, f"DEMO / {label}", None
                wx.wind_x = 0.6 if kind == "storm" else 0.3
                os.environ["FACE_CLOUD"] = str(cloud)
                os.environ["FACE_METEORS"] = str(extra.get("meteors", 0.3))
                space.kp = extra.get("kp", 0.0)
                space.fake_iss = "iss" in extra
                if space.fake_iss:
                    space.fake_cross = extra["iss"]
                    space.fake_t0 = time.monotonic() - 1
                sky.MOON_ELONG = None
            # ease in/out so time-lapses glide between scenes
            e = f * f * (3 - 2 * f)
            sky.TIME_OFFSET = a + (b - a) * e - time.time()
            if extra.get("phase"):
                sky.MOON_ELONG = 2 * math.pi * f        # new -> full -> new
            time.sleep(0.05)


class Face:
    def __init__(self):
        self.root = tk.Tk(className="hey-claude-face")
        self.root.title("Hey Claude Face")
        # nemo-desktop paints an opaque background, so we sit above it and leave
        # a strip on the left uncovered where it lays out the desktop icons.
        self.X = LEFT_MARGIN
        self.W = self.root.winfo_screenwidth() - self.X
        self.H = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}+{self.X}+0")
        self.root.configure(bg=hexcol(BG))
        try:
            # Desktop-type window: no decorations, sits below every normal window.
            self.root.attributes("-type", "desktop")
        except tk.TclError:
            self.root.overrideredirect(True)
            self.root.lower()
        self.c = tk.Canvas(self.root, width=self.W, height=self.H, bg=hexcol(BG),
                           highlightthickness=0, bd=0)
        self.c.pack(fill="both", expand=True)
        self.u = min(self.W, self.H) / 1000  # layout unit
        self.sensors = Sensors()
        self.dodger = WindowDodger(self.W, self.H, self.H * 0.5, self.H * 0.14, self.H * 0.72)
        self.head_x = self.W / 2
        self.t0 = time.monotonic()
        self.cur_state = "idle"
        self.blend = 0.0                 # 0..1 transition progress
        self.prim = STATE_COLORS["idle"][0]
        self.sec = STATE_COLORS["idle"][1]
        self.bright = STATE_COLORS["idle"][2]
        self.next_blink = time.monotonic() + 2
        self.blink_t = -1.0
        self.look = [0.0, 0.0]
        self.look_target = [0.0, 0.0]
        self.next_look = 0.0
        self.mouth_hist = [0.0] * 48
        self.weather = Weather()
        self.vitals = sky.Vitals()
        self.grid_phase, self.last_frame = 0.0, None
        self._draw_static()
        if os.environ.get("FACE_SHOWCASE") == "1":
            self.showcase = Showcase(self)
        self.root.after(100, self.tick)

    # glow = several strokes from wide/dim to thin/bright
    def glow_line(self, pts, rgb, width, k=1.0, smooth=False, closed=False):
        if closed:
            pts = list(pts) + list(pts[:2])
        for w_mul, b_mul in ((5.0, 0.12), (2.6, 0.3), (1.4, 0.65), (0.6, 1.0)):
            self.c.create_line(*pts, fill=hexcol(rgb, k * b_mul), width=max(1, width * w_mul),
                               smooth=smooth, capstyle="round", joinstyle="round", tags="dyn")

    def glow_arc(self, box, start, extent, rgb, width, k=1.0):
        for w_mul, b_mul in ((4.0, 0.12), (2.0, 0.35), (0.7, 1.0)):
            self.c.create_arc(*box, start=start, extent=extent, style="arc",
                              outline=hexcol(rgb, k * b_mul), width=max(1, width * w_mul), tags="dyn")

    def _draw_static(self):
        W, H, u = self.W, self.H, self.u
        # living sky (sun, moon, stars, aurora, meteors, ISS) sits behind everything
        self.sky = sky.Sky(self, Weather.location)
        self.sky.build_back()
        # synthwave perspective grid on the lower third
        horizon = H * 0.72
        for i in range(-24, 25):
            x_far = W / 2 + i * W * 0.02
            x_near = W / 2 + i * W * 0.16
            self.c.create_line(x_far, horizon, x_near, H, fill="#3a0c16", width=max(1, 2 * u), tags="grid")
        self.grid_rows = []
        for i in range(14):
            line = self.c.create_line(0, horizon, W, horizon, fill="#160408", width=max(1, 2 * u), tags="grid")
            self.grid_rows.append(line)
        # glowing horizon: persistent items created before the head, so they stay behind it
        self.horizon_lines = [
            (self.c.create_line(0, horizon, W, horizon, fill="#000000", width=max(1, 2 * u * w_mul),
                                capstyle="round", tags="grid"), b_mul)
            for w_mul, b_mul in ((5.0, 0.12), (2.6, 0.3), (1.4, 0.65), (0.6, 1.0))]
        # weather particles: created here so they sit above the grid but behind the head
        self._build_weather()
        # scanlines
        step = max(3, int(4 * u))
        for y in range(0, H, step):
            self.c.create_line(0, y, W, y, fill="#060206", tags="scan")
        self.label = self.c.create_text(W / 2, H * 0.9, text="", font=("Monospace", int(26 * u), "bold"),
                                        fill="#ffffff", tags="hud")
        self.sub = self.c.create_text(W / 2, H * 0.94, text="", font=("Monospace", int(13 * u)),
                                      fill="#888888", tags="hud")

    def tick(self):
        start = time.monotonic()
        try:
            self.frame(start - self.t0)
        except Exception as e:  # never die on a drawing glitch
            print("frame error:", e, flush=True)
        end = time.monotonic()
        self._perf(start, end)
        delay = max(5, int(1000 / FPS - (end - start) * 1000))
        self.root.after(delay, self.tick)

    def _perf(self, start, end):
        """Every 5 s append fps / frame cost per state to logs/perf.log (FACE_PERF=0 disables)."""
        if os.environ.get("FACE_PERF", "1") == "0":
            return
        if not hasattr(self, "pf"):
            self.pf = {"t0": end, "n": 0, "draw": [], "gaps": [], "last": start, "states": {},
                       "cpu0": time.process_time()}
        pf = self.pf
        pf["gaps"].append((start - pf["last"]) * 1000)
        pf["last"] = start
        pf["draw"].append((end - start) * 1000)
        pf["n"] += 1
        st = self.cur_state + ("/fem" if getattr(self, "fem", 0) > 0.5 else "")
        pf["states"][st] = pf["states"].get(st, 0) + 1
        span = end - pf["t0"]
        if span < 5:
            return
        d, g = sorted(pf["draw"]), sorted(pf["gaps"][1:] or [0])
        cpu = (time.process_time() - pf["cpu0"]) / span * 100
        line = (f"{time.strftime('%H:%M:%S')} state={max(pf['states'], key=pf['states'].get):14s} "
                f"fps={pf['n'] / span:5.1f}/{FPS} draw_ms avg={sum(d) / len(d):5.1f} p95={d[int(len(d) * .95)]:5.1f} "
                f"max={d[-1]:5.1f} gap_ms max={g[-1]:6.1f} slow_frames={sum(x > 1500 / FPS for x in g)} "
                f"cpu={cpu:4.0f}% items={len(self.c.find_all())}")
        try:
            if LOG_PERF.exists() and LOG_PERF.stat().st_size > 2_000_000:   # keep one old copy
                LOG_PERF.replace(LOG_PERF.with_name(LOG_PERF.name + ".1"))
            with open(LOG_PERF, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        self.pf = {"t0": end, "n": 0, "draw": [], "gaps": [], "last": start, "states": {},
                   "cpu0": time.process_time()}

    def frame(self, t):
        W, H, u = self.W, self.H, self.u
        now = time.monotonic()
        state = self.sensors.state()
        if state != self.cur_state:
            self.cur_state = state
        p, s, b = STATE_COLORS[state]
        self.prim = lerp_rgb(self.prim, p, 0.12)
        self.sec = lerp_rgb(self.sec, s, 0.12)
        self.bright = lerp(self.bright, b, 0.08)
        prim, sec, k = self.prim, self.sec, self.bright
        fem_target = 1.0 if self.sensors.feminine() else 0.0
        if state == "idle" and getattr(self, "presence", 0.0) > 0.02:
            fem_target = getattr(self, "fem", 0.0)       # don't morph while fading out
        self.fem_target = fem_target
        self.fem = lerp(getattr(self, "fem", 0.0), fem_target, 0.05)
        if self.fem < 0.002:
            self.fem = 0.0
        # female-only features (hair, undercut, lips, lashes) follow fem on the way in but
        # fade out in ~0.5 s when switching back, instead of riding the slow head morph
        if fem_target > 0.5:
            self.fem_x = self.fem
        else:
            self.fem_x = min(self.fem, getattr(self, "fem_x", 0.0) * 0.7)
            if self.fem_x < 0.01:
                self.fem_x = 0.0
        fem = self.fem
        tint = lambda c: (c[0] * 0.5, c[1] * 0.3, min(255, c[2] * 0.4 + 150))   # red -> deep purple
        prim, sec = lerp_rgb(prim, tint(prim), fem), lerp_rgb(sec, tint(sec), fem)
        breathe = 0.85 + 0.15 * math.sin(t * 1.3)
        if state == "idle":
            k *= breathe

        self.c.delete("dyn")
        try:
            self._draw_weather(t, prim, sec, k)
        except Exception as e:
            print("weather draw error:", e, flush=True)
        fk = 1 + 2.5 * self.flash          # lightning lights up the floor
        pk = 1.0
        try:
            pk = self.sky.update(t, prim, sec, k, self.flash)   # low battery dims/flickers the scene
        except Exception as e:
            print("sky draw error:", e, flush=True)
        k *= pk
        # vital signs: CPU load speeds up the floor, a hot CPU tints the horizon orange
        dt = min(0.1, now - (self.last_frame or now))
        self.last_frame = now
        self.grid_phase = (self.grid_phase + dt * 0.6 * self.vitals.grid_speed()) % 1
        sec_floor = lerp_rgb(sec, (255, 130, 25), self.vitals.heat())

        # moving grid rows
        horizon = H * 0.72
        for i, line in enumerate(self.grid_rows):
            f = ((i + self.grid_phase) / len(self.grid_rows))
            y = horizon + (H - horizon) * f ** 2.2
            self.c.coords(line, 0, y, W, y)
            self.c.itemconfigure(line, fill=hexcol(sec_floor, 0.3 * max(k, 0.6 * pk) * (0.3 + f) * fk))
        # glowing horizon
        hk = 0.5 * max(k, 0.6 * pk) * breathe * fk
        for line, b_mul in self.horizon_lines:
            self.c.itemconfigure(line, fill=hexcol(sec_floor, hk * b_mul))

        # The face only exists while a voice session is active; it fades in/out.
        target = 0.0 if state == "idle" else 1.0
        self.presence = lerp(getattr(self, "presence", 0.0), target, 0.08 if target else 0.04)
        hud_k = k * max(self.presence, 0.4)
        if self.presence < 0.02:
            self.c.itemconfigure("head", state="hidden")
            self.mask_cache = None       # everything was hidden by tag; force a full refresh
            self.hair_cache = None
            self._draw_hud(state, t, prim, sec, hud_k)
            return
        k *= self.presence

        # eyes: blink + gaze
        if now > self.next_blink and self.blink_t < 0:
            self.blink_t = 0.0
        openness = 1.0
        if self.blink_t >= 0:
            self.blink_t += 1 / FPS
            openness = abs(1 - self.blink_t / 0.09) if self.blink_t < 0.18 else 1.0
            if self.blink_t >= 0.18:
                self.blink_t = -1.0
                self.next_blink = now + random.uniform(2.5, 6.0)
        if now > self.next_look:
            if state == "thinking":
                self.look_target = [random.choice([-1, 1]) * random.uniform(0.4, 1), random.uniform(-0.3, 1)]
                self.next_look = now + random.uniform(0.4, 1.1)
            else:
                self.look_target = [random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2)]
                self.next_look = now + random.uniform(1.5, 4)
        self.look[0] = lerp(self.look[0], self.look_target[0], 0.2)
        self.look[1] = lerp(self.look[1], self.look_target[1], 0.2)
        lvl = self.sensors.mouth_level() if state == "speaking" else 0.0
        self.jaw = lerp(getattr(self, "jaw", 0.0), lvl, 0.5)
        self._draw_head(state, t, prim, sec, k, openness)
        self._draw_hud(state, t, prim, sec, hud_k)

    # ---- 3D hologram head ----------------------------------------------------------
    def _project(self, P):
        """P: (...,3) world points -> screen x, y and depth (0 back .. 1 front)."""
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        cyw, syw, cp, sp = self.rot
        x1 = x * cyw + z * syw
        z1 = -x * syw + z * cyw
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        f = self.cam / (self.cam - z2)
        sx = self.hcx + self.scale * x1 * f
        sy = self.hcy - self.scale * y2 * f
        return sx, sy, np.clip((z2 + 1.0) / 2.0, 0, 1)

    def _polyline(self, sx, sy, d, prim, sec, k, width, closed=False, glow=True, boost=None):
        """Draw a projected polyline split into runs of similar depth (dim at the back)."""
        if closed:
            sx, sy, d = np.append(sx, sx[0]), np.append(sy, sy[0]), np.append(d, d[0])
            if boost is not None:
                boost = np.append(boost, boost[0])
        n = len(sx)
        dm = (d[:-1] + d[1:]) / 2
        if boost is not None:
            dm = np.minimum(1.3, dm + (boost[:-1] + boost[1:]) / 2)
        bucket = np.minimum(5, (dm * 5).astype(int))
        i = 0
        while i < n - 1:
            j = i
            while j + 1 < n - 1 and bucket[j + 1] == bucket[i]:
                j += 1
            b = bucket[i] / 5
            col = lerp_rgb(sec, prim, min(1, b))
            kk = k * (0.12 + 0.88 * b ** 1.6)
            pts = np.column_stack((sx[i:j + 2], sy[i:j + 2])).ravel().tolist()
            if glow and b >= 0.6:
                self.c.create_line(*pts, fill=hexcol(col, kk * 0.22), width=max(1, width * 4),
                                   capstyle="round", joinstyle="round", tags="dyn")
            self.c.create_line(*pts, fill=hexcol(col, kk), width=max(1, width),
                               capstyle="round", joinstyle="round", tags="dyn")
            i = j + 1

    # head silhouette profiles, top (y=1) to chin (y=-1)
    _PY = np.array([1.0, 0.93, 0.8, 0.6, 0.35, 0.15, -0.1, -0.35, -0.6, -0.8, -0.93, -1.0])
    _PW = np.array([0.0, 0.36, 0.56, 0.68, 0.72, 0.72, 0.70, 0.66, 0.58, 0.47, 0.33, 0.20])   # half width
    _PF = np.array([0.0, 0.42, 0.66, 0.80, 0.86, 0.86, 0.84, 0.80, 0.76, 0.70, 0.60, 0.50])   # front depth
    _PB = np.array([0.0, 0.55, 0.86, 1.02, 1.06, 1.0, 0.86, 0.62, 0.38, 0.24, 0.18, 0.16])   # back depth
    # feminine variant: narrower jaw, pointed chin, rounder skull
    _PW_F = np.array([0.0, 0.36, 0.55, 0.66, 0.69, 0.68, 0.65, 0.59, 0.49, 0.37, 0.25, 0.14])
    _PF_F = np.array([0.0, 0.42, 0.66, 0.80, 0.85, 0.85, 0.83, 0.79, 0.74, 0.67, 0.57, 0.47])
    _PB_F = np.array([0.0, 0.60, 0.92, 1.07, 1.10, 1.04, 0.90, 0.66, 0.40, 0.25, 0.18, 0.15])

    @classmethod
    def _surface(cls, y, phi, jaw=0.0, fem=0.0):
        """Head surface point at height y and angle phi around the head (0 = facing us).
        fem 0..1 morphs towards the feminine head."""
        y = np.asarray(y, dtype=float) + np.zeros_like(phi, dtype=float)
        yi = np.clip(y, -1, 1)
        w = np.interp(-yi, -cls._PY, lerp(cls._PW, cls._PW_F, fem))
        zf = np.interp(-yi, -cls._PY, lerp(cls._PF, cls._PF_F, fem))
        zb = np.interp(-yi, -cls._PY, lerp(cls._PB, cls._PB_F, fem))
        sp, cp = np.sin(phi), np.cos(phi)
        x = w * np.sign(sp) * np.abs(sp) ** 0.9
        z = np.where(cp > 0, zf, zb) * np.sign(cp) * np.abs(cp) ** 0.75   # flatter face
        front = np.clip(cp * 2.5, 0, 1)
        g = lambda cx, cy, sx, sy: np.exp(-((x - cx) / sx) ** 2 - ((y - cy) / sy) ** 2)
        m = 1 - fem
        z = z + front * (lerp(0.16, 0.12, fem) * g(0, -0.13, lerp(0.07, 0.055, fem), 0.17)
                         + 0.05 * m * g(0, -0.26, 0.13, 0.05)                                # nose, nostrils
                         + lerp(0.09, 0.035, fem) * g(0, 0.27, 0.5, 0.05) + 0.03 * g(0, 0.22, 0.1, 0.05)                                   # brow ridge
                         - 0.11 * g(-0.27, 0.15, 0.15, 0.08) - 0.11 * g(0.27, 0.15, 0.15, 0.08)  # sockets
                         + 0.07 * g(-0.4, -0.08, 0.1, 0.08) + 0.07 * g(0.4, -0.08, 0.1, 0.08) - 0.04 * g(-0.3, -0.3, 0.1, 0.12) - 0.04 * g(0.3, -0.3, 0.1, 0.12)  # cheekbones
                         + 0.04 * fem * g(-0.38, 0.0, 0.1, 0.07) + 0.04 * fem * g(0.38, 0.0, 0.1, 0.07)  # high cheekbones
                         + lerp(0.05, 0.075, fem) * g(0, -0.45, lerp(0.17, 0.15, fem), lerp(0.07, 0.085, fem))  # lips
                         + lerp(0.06, 0.035, fem) * g(0, -0.82, 0.15, 0.1))                                 # chin
        if jaw:  # speech: everything under the mouth line drops
            wj = front * np.clip((-0.46 - y) / 0.05, 0, 1) * np.clip(1 - np.abs(x) / 0.6, 0, 1)
            y = y - jaw * 0.12 * wj
        return np.stack((x, y, z), axis=-1)

    def _build_mask(self):
        """Low-poly faceted head: jittered grid -> triangles, items created once, back to front."""
        rng = np.random.default_rng(7)
        NR, NC = 17, 26
        y = np.linspace(0.97, -0.97, NR)[:, None] + np.zeros((NR, NC))
        phi = np.linspace(-np.pi, np.pi, NC, endpoint=False)[None, :] + np.zeros((NR, NC))
        # jitter mirrored left/right so the facets are symmetric about the face's centre line
        jy = rng.uniform(-0.025, 0.025, (NR - 2, NC))
        jp = rng.uniform(-0.05, 0.05, (NR, NC))
        for c in range(NC):
            m = (NC - c) % NC                   # column at -phi
            if m == c:
                jp[:, c] = 0
            elif m < c:
                jy[:, c], jp[:, c] = jy[:, m], -jp[:, m]
        y[1:-1] += jy
        phi += jp
        self.MY, self.MPHI = y, phi
        idx = np.arange(NR * NC).reshape(NR, NC)
        tris = []
        for r in range(NR - 1):
            for c in range(NC):
                a, b = idx[r, c], idx[r, (c + 1) % NC]
                d, e = idx[r + 1, c], idx[r + 1, (c + 1) % NC]
                cm = (NC - c - 1) % NC               # mirror quad; flip its diagonal
                diag = (r + c) % 2 if c <= cm else 1 - (r + cm) % 2
                if diag:
                    tris += [(a, b, e), (a, e, d)]
                else:
                    tris += [(a, b, d), (b, e, d)]
        T = np.array(tris)
        V = self._surface(y, phi).reshape(-1, 3)
        order = np.argsort(V[T].mean(1)[:, 2])           # back to front (neutral pose)
        self.T = T[order]
        cen = V[self.T].mean(1)
        # how close each facet is to an eye (red light spill)
        self.eye_w = np.exp(-((np.abs(cen[:, 0]) - 0.27) / 0.16) ** 2 - ((cen[:, 1] - 0.15) / 0.12) ** 2) \
            * (cen[:, 2] > 0)
        self.polys = [self.c.create_polygon(0, 0, 0, 0, 0, 0, fill="", outline="", width=max(1, 1.2 * self.u),
                                            tags="head", state="hidden") for _ in range(len(self.T))]
        self.poly_style = [None] * len(self.T)
        self.scale = self.H * 0.27
        self.cam = 5.0

    def _cam(self, P):
        """World -> camera space (yaw then pitch)."""
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        cyw, syw, cp, sp = self.rot
        x1 = x * cyw + z * syw
        z1 = -x * syw + z * cyw
        return np.stack((x1, y * cp - z1 * sp, y * sp + z1 * cp), -1)

    def _glow_poly(self, P, rgb, k, width, closed=False, layers=((5, 0.1), (2.4, 0.3), (1, 1.0)),
                   base=(16, 13, 20)):
        """Fake bloom: wide strokes blended towards the mask colour underneath, then a bright core."""
        px, py, _ = self._project(P)
        pts = np.column_stack((px, py))
        if closed:
            pts = np.vstack((pts, pts[:1]))
        pts = pts.ravel().tolist()
        for wm, bm in layers:
            col = lerp_rgb(base, rgb, min(1.0, bm * k))
            self.c.create_line(*pts, fill=hexcol(col, 1.0 if k > 0.05 else k * 20), width=max(1, width * wm),
                               capstyle="round", joinstyle="round", tags="dyn")

    # ---- Tron-style decoration --------------------------------------------------------
    # Circuit traces / face-paint accents were tried and removed at the user's request
    # (too busy). Add (y, phi) waypoint lists here to bring them back.
    _TRACES = []
    _ACCENTS = []

    @staticmethod
    def _densify(pts, per=8):
        out = []
        for (y0, p0), (y1, p1) in zip(pts, pts[1:]):
            for i in range(per):
                f = i / per
                out.append((y0 + (y1 - y0) * f, p0 + (p1 - p0) * f))
        out.append(pts[-1])
        return np.array(out)

    def _visible(self, P):
        """True where a surface point faces the camera (outward ~ horizontal from the axis)."""
        C = self._cam(P)
        O = self._cam(P * np.array([1.0, 0.25, 1.0]))
        view = np.array([0, 0, self.cam]) - C
        return np.einsum("ij,ij->i", O, view) > 0.15 * np.linalg.norm(view, axis=1)

    def _trace(self, P, rgb, k, width, layers=((4, 0.12), (2, 0.35), (0.8, 1.0))):
        vis = self._visible(P)
        i, n = 0, len(P)
        while i < n:
            if not vis[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and vis[j + 1]:
                j += 1
            if j > i:
                self._glow_poly(P[i:j + 1], rgb, k, width, layers=layers)
            i = j + 1

    def _draw_tron(self, state, t, prim, sec, k):
        u = self.u
        if not hasattr(self, "trace_pts"):
            self.trace_pts = []
            for path in self._TRACES:
                d = self._densify(path)
                for side in (-1, 1):
                    self.trace_pts.append((d[:, 0], d[:, 1] * side, random.random()))
            self.accent_pts = []
            for path in self._ACCENTS:
                d = self._densify(path, 6)
                for side in (-1, 1):
                    self.accent_pts.append((d[:, 0], d[:, 1] * side))
        speed = {"thinking": 0.9, "speaking": 0.5, "listening": 0.3}.get(state, 0.0)
        lvl = self.jaw if state == "speaking" else 0.0
        hot = (255, 255, 255)
        for ty, tp, off in self.trace_pts:
            P = self._surface(ty, tp) * 1.008
            self._trace(P, prim, k * (0.3 + 0.25 * lvl), 1.2 * u)
            # light pulse running along the trace
            n = len(P)
            pos = ((t * speed + off) % 1) * (n + 4) - 2
            a, b = int(max(0, pos - 2)), int(min(n, pos + 2))
            if b - a >= 2 and speed:
                self._trace(P[a:b], hot, k * 0.6, 1.8 * u, layers=((3, 0.3), (1, 1.0)))
            # node at the end of each trace
            end = P[-1:]
            if self._visible(end)[0]:
                px, py, _ = self._project(end)
                r = 4 * u
                self.c.create_oval(px[0] - r, py[0] - r, px[0] + r, py[0] + r, outline=hexcol(prim, k),
                                   width=max(1, 1.5 * u), fill=hexcol((16, 13, 20), 1), tags="dyn")
        for ay, ap in self.accent_pts:
            P = self._surface(ay, ap) * 1.01
            self._trace(P, prim, k * 0.75, 1.8 * u, layers=((4, 0.1), (2, 0.3), (1, 1.0)))
        # temple discs with rotating segments
        if self.fem > 0.98:
            return                          # temple discs are under her hair
        k *= 1 - self.fem
        spin = t * (3.0 if state == "thinking" else 0.6)
        a = np.linspace(0, 2 * np.pi, 32)
        for side in (-1, 1):
            for rad, kk in ((1.0, 0.9), (0.6, 0.5)):
                P = self._surface(0.12 + 0.13 * rad * np.sin(a), side * (1.42 + 0.17 * rad * np.cos(a)), 0.0, self.fem) * 1.01
                self._trace(P, prim, k * kk, 1.6 * u)
            for j in range(3):
                seg = a[:8] + spin * side + j * 2.1
                P = self._surface(0.12 + 0.105 * np.sin(seg), side * (1.42 + 0.137 * np.cos(seg)), 0.0, self.fem) * 1.012
                self._trace(P, hot if state != "idle" else prim, k * (0.6 + 0.4 * lvl), 2.2 * u,
                            layers=((3, 0.3), (1, 1.0)))

    def _draw_head(self, state, t, prim, sec, k, openness):
        W, H, u = self.W, self.H, self.u
        if not hasattr(self, "T"):
            self._build_mask()
        self.head_x = lerp(self.head_x, self.dodger.target_x - self.X, 0.035)   # glide
        self.hcx, self.hcy = self.head_x, H * 0.42 + math.sin(t * 0.6) * 4 * u
        lx, ly = self.look
        yaw = 0.16 * math.sin(t * 0.17) + lx * 0.18
        foc = self.dodger.focus
        if foc and state != "listening":       # look at the focused window
            fy_ = (foc[0] - self.X - self.head_x) / W
            fp_ = (foc[1] - H * 0.42) / H
            tgt = (max(-0.65, min(0.65, fy_ * 1.6)), max(-0.25, min(0.25, fp_ * 0.6)))
        else:                                  # otherwise face inwards, towards screen centre
            tgt = (-(self.head_x - W / 2) / W * 0.9, 0.0)
        fo = getattr(self, "focus_turn", (0.0, 0.0))
        self.focus_turn = (lerp(fo[0], tgt[0], 0.05), lerp(fo[1], tgt[1], 0.05))
        yaw += self.focus_turn[0]
        if state == "listening":
            yaw *= 0.3                       # stares straight at you
        pitch = 0.10 + 0.03 * math.sin(t * 0.23) - ly * 0.05   # chin slightly down: looming
        pitch += self.focus_turn[1]
        self.rot = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch))
        if random.random() < 0.01:          # rare glitch tear
            self.hcx += random.uniform(-14, 14) * u

        # ---- faceted mask
        fem = self.fem
        V = self._surface(self.MY, self.MPHI, self.jaw * 0.6, fem).reshape(-1, 3)
        C = self._cam(V)
        A, B, D = C[self.T[:, 0]], C[self.T[:, 1]], C[self.T[:, 2]]
        n = np.cross(B - A, D - A)
        cen = (A + B + D) / 3
        n *= np.sign(np.einsum("ij,ij->i", n, cen))[:, None]       # point outwards
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        view = np.array([0, 0, self.cam]) - cen
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", n, view)
        L = np.array([-0.45, 0.65, 0.6]); L /= np.linalg.norm(L)
        diff = np.clip(n @ L, 0, 1)
        under = np.clip(n @ np.array([0.0, -0.8, 0.6]), 0, 1)     # red under-light
        rim = np.clip(1 - facing, 0, 1) ** 2
        f = self.cam / (self.cam - C[..., 2])
        SX = self.hcx + self.scale * C[..., 0] * f
        SY = self.hcy - self.scale * C[..., 1] * f
        eye_glow = 0.35 + 0.65 * max(0.0, openness) if state != "idle" else 0.25
        cen_y = V[self.T].mean(1)[:, 1]
        if state == "thinking":   # bright scan bands racing down the face, eyes pulsing
            band = sum(np.exp(-((cen_y - (1.2 - ((t * 0.9 + off) % 1) * 2.6)) / 0.09) ** 2) for off in (0, 0.5))
            eye_glow *= 0.55 + 0.45 * abs(math.sin(t * 5))
        else:                     # faint slow scan otherwise
            band = 0.15 * np.exp(-((cen_y - (1.2 - ((t * 0.25) % 1) * 3.0)) / 0.05) ** 2)
        # vectorised shading, then one batched Tcl script for all facets
        q = lambda v: np.round(v * 16) / 16                           # quantise -> fewer reconfigs
        dl, ud, rm = q(diff)[:, None], q(under)[:, None], q(rim)[:, None]
        ew, bd = q(self.eye_w * eye_glow)[:, None], q(band)[:, None]
        P_, S_ = np.array(prim), np.array(sec)
        base = np.array([9, 8, 13]) + np.array([38, 38, 46]) * dl ** 2   # obsidian + cold key light
        fill = base + S_ * 0.10 * ud + P_ * 0.22 * ew + P_ * 0.4 * bd
        tt = np.minimum(1, 0.05 + 0.7 * rm ** 1.3 + 1.0 * bd)
        edge = fill + (P_ - fill) * tt
        pts = np.stack((SX[self.T], SY[self.T]), -1).reshape(-1, 6)
        self._apply_facets(self.polys, "mask_cache", facing >= 0.0, pts, fill, edge, k)

        self._draw_tron(state, t, prim, sec, k)

        # ---- mouth: a thin slit, a dark groove at rest; it ignites when speaking
        lvl = self.jaw if state == "speaking" else 0.0
        n = 13
        ms = np.linspace(-0.19, 0.19, n)
        env = np.sin(np.linspace(0, np.pi, n)) ** 0.7
        wave = np.abs(np.sin(np.arange(n) * 1.3 + t * 22)) * 0.5 + 0.5
        gap = 0.004 + 0.022 * lvl * env * wave
        top = self._surface(-0.455 + gap * 0.3, ms, 0.0, fem) * (1.012 + 0.012 * fem)
        bot = self._surface(-0.455 - gap, ms[::-1], self.jaw * 0.3, fem) * (1.012 + 0.012 * fem)
        slit = np.vstack((top, bot))
        if lvl > 0.04:
            self._glow_poly(slit, prim, k * min(1, 0.4 + lvl), 2.2 * u, closed=True,
                            layers=((7, 0.1), (3.5, 0.3), (1.4, 0.8)))
            px, py, _ = self._project(slit)
            self.c.create_polygon(*np.column_stack((px, py)).ravel().tolist(),
                                  fill=hexcol(lerp_rgb(prim, (255, 255, 255), 0.35), k * min(1, 0.5 + lvl)),
                                  outline="", tags="dyn")
        else:
            idle_k = 0.18 + (0.12 * (0.5 + 0.5 * math.sin(t * 2)) if state == "listening" else 0)
            self._glow_poly(slit[:n], prim, k * idle_k, 1.3 * u, layers=((3, 0.3), (1, 1.0)))
        # (forehead crest line removed at the user's request; the chip replaces it)
        self._draw_chip(state, t, prim, sec, k)
        # female-only features use the fast-fading fem_x (see draw())
        hair_k = fx = getattr(self, "fem_x", 0.0)
        if fx > 0.02:
            self._draw_lips(t, prim, sec, k * fx, lvl)
        if hair_k > 0.02:
            self._draw_hair_volume(prim, sec, k * hair_k, t)
            self._draw_undercut(state, t, prim, sec, k * hair_k)
        else:
            self._hide_hair()

        # ---- eyes: narrow slanted slits, inner corner low
        e_open = max(0.0, openness) * (1.25 if state == "listening" else 1.0)
        if state == "idle":
            e_open *= 0.45
        hot = (255, 255, 255)
        for side in (-1, 1):
            th = 0.022 * e_open + 0.003
            ey = np.array([0.120, 0.150 + th, 0.188 + th * 0.4, 0.172 - th * 0.3, 0.130 - th * 0.6])
            ep = side * np.array([0.17, 0.33, 0.52, 0.40, 0.25])
            slit = self._surface(ey, ep, 0.0, fem) * 1.015
            self._glow_poly(slit, sec, k * eye_glow, 3.2 * u, closed=True,
                            layers=((9, 0.08), (5, 0.18), (2.2, 0.55), (1, 1.0)))
            px, py, _ = self._project(slit)
            self.c.create_polygon(*np.column_stack((px, py)).ravel().tolist(),
                                  fill=hexcol(lerp_rgb(prim, hot, 0.4), k * eye_glow), outline="", tags="dyn")
            # white-hot core that slides with the gaze
            if e_open > 0.25:
                gp = side * (0.34 + lx * 0.07 * side)
                core = self._surface(np.array([0.155 + 0.03 * (abs(gp) - 0.34) * 2] * 2),
                                     np.array([gp - 0.05 * side, gp + 0.05 * side]), 0.0, fem) * 1.02
                self._glow_poly(core, hot, k * eye_glow, 2.5 * u, layers=((3, 0.35), (1, 1.0)))
            if fx > 0.02:   # lashes flicking up from the outer corner
                for ly0, lp0, ly1, lp1 in ((0.186, 0.50, 0.225, 0.60), (0.180, 0.45, 0.228, 0.53),
                                           (0.172, 0.40, 0.215, 0.46)):
                    yy = np.array([ly0 + th * 0.4, ly1 + th * 0.4])
                    P = self._surface(yy, side * np.array([lp0, lp1]), 0.0, fem) * 1.02
                    self._glow_poly(P, sec, k * fx * eye_glow, 1.4 * u, layers=((3, 0.25), (1, 1.0)))

    def _draw_lips(self, t, prim, sec, k, lvl):
        """Cupid's-bow upper lip and a soft lower-lip curve around the mouth slit."""
        u, fem = self.u, self.fem
        a = np.linspace(-1, 1, 15)
        bow = 0.012 * np.exp(-((np.abs(a) - 0.35) / 0.25) ** 2) - 0.008 * np.exp(-(a / 0.15) ** 2)
        upper = self._surface(-0.428 + bow - 0.02 * a ** 2, a * 0.2, 0.0, fem) * 1.03
        lower = self._surface(-0.478 - 0.028 * (1 - a ** 2) - 0.03 * lvl, a * 0.18, self.jaw * 0.3, fem) * 1.03
        for P in (upper, lower):
            self._glow_poly(P, prim, k * (0.45 + 0.4 * lvl), 1.6 * u, layers=((4, 0.12), (2, 0.3), (1, 1.0)))

    def _build_hair(self):
        """Solid faceted hair volume: a shell over the skull with a side-swept hairline,
        open at the face, draping down behind the neck. Two item pools: facets seen from
        inside (behind the head) sit under the mask, outside facets sit on top of it."""
        rng = np.random.default_rng(5)
        NR, NC = 11, 30
        phi = np.linspace(-np.pi, np.pi, NC, endpoint=False) + np.pi / NC
        # cyberpunk undercut: right side (phi > 0) shaved high, left side swept long over the
        # ear, short at the nape; ragged ends
        ap = np.abs(phi)
        ends = np.where(phi > 0, 0.46, 0.08 - 0.1 * np.clip((ap - 0.9) / 0.6, 0, 1))
        ends = np.where(ap > 2.2, np.minimum(ends, 0.0) + 0.1 * (phi > 0), ends)
        ends = ends - 0.07 * rng.random(NC)
        s_ = np.linspace(0, 1, NR)[:, None] ** 1.1
        self.HY = 1.0 - (1.0 - ends[None, :]) * s_
        # swept towards the long side, more so further down
        self.HPHI = phi[None, :] - 0.22 * s_ + rng.uniform(-0.04, 0.04, (NR, NC))
        # spikes: random outward pushes, strongest at the crown
        spike = rng.random((NR, NC)) ** 2 * (0.05 + 0.12 * (1 - s_)) * (np.arange(NR)[:, None] % 2 == 1)
        spike[0] = 0
        self.HPUFF = (1.0 + spike).ravel()
        self.HOFF = rng.uniform(0, 6, (NR, NC))
        idx = np.arange(NR * NC).reshape(NR, NC)
        tris = []
        for r in range(NR - 1):
            for c in range(NC):
                a, b = idx[r, c], idx[r, (c + 1) % NC]
                d, e = idx[r + 1, c], idx[r + 1, (c + 1) % NC]
                tris += [(a, b, e), (a, e, d)] if (r + c) % 2 else [(a, b, d), (b, e, d)]
        T = np.array(tris)
        Y, PH = self.HY.ravel(), self.HPHI.ravel()
        yc, pc = Y[T].mean(1), PH[T].mean(1)
        hairline = 0.6 + 0.22 * pc                     # fringe swept down over the left brow
        keep = (np.abs(pc) > 0.92) | (yc > hairline)
        T = T[keep]
        V = self._hair_points(Y, PH, 0.0, 1.0, self.HOFF.ravel())
        order = np.argsort(V[T].mean(1)[:, 2])
        self.HT = T[order]
        mk = lambda tag: [self.c.create_polygon(0, 0, 0, 0, 0, 0, fill="", outline="", width=max(1, self.u),
                                                tags=("head", tag), state="hidden") for _ in range(len(self.HT))]
        self.hair_front = mk("hairfront")
        self.hair_style = [None] * len(self.HT)

    def _draw_hair_volume(self, prim, sec, k, t):
        if not hasattr(self, "HT"):
            self._build_hair()
        V = self._hair_points(self.HY.ravel(), self.HPHI.ravel(), t, self.fem, self.HOFF.ravel())
        V = V * self.HPUFF[:, None]
        C = self._cam(V)
        A, B, D = C[self.HT[:, 0]], C[self.HT[:, 1]], C[self.HT[:, 2]]
        cen = (A + B + D) / 3
        n = np.cross(B - A, D - A)
        Vc = V[self.HT].mean(1)
        radial = self._cam(Vc * np.array([1.0, 0.0, 1.0]) + np.array([0.0, 0.0, 0.0])) \
            + self._cam(np.column_stack((np.zeros(len(Vc)), np.maximum(Vc[:, 1] - 0.3, 0), np.zeros(len(Vc)))))
        n *= np.sign(np.einsum("ij,ij->i", n, radial) + 1e-9)[:, None]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        view = np.array([0, 0, self.cam]) - cen
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", n, view)
        L = np.array([-0.45, 0.65, 0.6]); L /= np.linalg.norm(L)
        diff = np.clip(np.abs(n @ L), 0, 1)
        f = self.cam / (self.cam - C[..., 2])
        SX = self.hcx + self.scale * C[..., 0] * f
        SY = self.hcy - self.scale * C[..., 1] * f
        q = lambda v: np.round(v * 12) / 12
        rim = q(np.clip(1 - np.abs(facing), 0, 1) ** 2)[:, None]
        dl = q(diff)[:, None]
        P_, S_ = np.array(prim), np.array(sec)
        fill = 14 + P_ * (0.07 + 0.2 * dl ** 2) + S_ * 0.05
        edge = fill + (P_ - fill) * (0.25 + 0.6 * rim)
        pts = np.stack((SX[self.HT], SY[self.HT]), -1).reshape(-1, 6)
        self._apply_facets(self.hair_front, "hair_cache", facing >= 0, pts, fill, edge, k)

    def _hide_hair(self):
        if hasattr(self, "HT") and getattr(self, "hair_cache", None) is not None:
            self.c.itemconfigure("hairfront", state="hidden")
            self.hair_cache = None

    def _apply_facets(self, items, cache_name, vis, pts, fill, edge, k):
        """Update many canvas polygons in one Tcl call. Colours are only re-sent for facets
        whose displayed (integer) colour changed; cache holds the last fill+edge, -1 = hidden."""
        kk = max(0.0, min(1.0, k))
        bg = np.array(BG)
        cols = np.concatenate([(bg + (fill - bg) * kk), (bg + (edge - bg) * kk)], 1)
        cols = np.clip(cols, 0, 255).astype(np.int16)
        cols[~vis] = -1
        cache = getattr(self, cache_name, None)
        if cache is None or len(cache) != len(items):
            cache = np.full((len(items), 6), -2, np.int16)
        changed = np.any(cols != cache, axis=1)
        if hasattr(self.c, "set_facets"):          # GPU renderer: hand over the arrays directly
            self.c.set_facets(items, vis, pts, cols, changed)
            setattr(self, cache_name, cols)
            return
        w = self.c._w
        out = []
        for i in np.flatnonzero(vis):
            p = pts[i]
            out.append(f"{w} coords {items[i]} {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {p[3]:.1f} {p[4]:.1f} {p[5]:.1f}")
        for i in np.flatnonzero(changed):
            c = cols[i]
            if c[0] < 0:
                out.append(f"{w} itemconfigure {items[i]} -state hidden")
            else:
                out.append(f"{w} itemconfigure {items[i]} -state normal "
                           f"-fill #{c[0]:02x}{c[1]:02x}{c[2]:02x} -outline #{c[3]:02x}{c[4]:02x}{c[5]:02x}")
        if out:
            self.c.tk.eval("\n".join(out))
        setattr(self, cache_name, cols)

    def _hair_points(self, y, phi, t, fem, off, shell=1.07):
        """Hair shell: hugs the skull, then drapes straight down past the jaw to the shoulders."""
        sway = 0.012 * np.sin(t * 1.1 + off + y * 2.5) * np.clip((0.3 - y) / 1.5, 0, 1)
        P = self._surface(np.clip(y, -1, 1), phi + sway, 0.0, fem) * shell
        P[:, 1] += 0.06 * np.clip(y, 0, 1) ** 2                 # a little volume on top
        r = np.hypot(P[:, 0], P[:, 2]) + 1e-6
        rmin = np.where(y < -9, (0.8 + 0.1 * np.clip(0.25 - y, 0, 1)) * shell / 1.07, 0.0)
        s = np.maximum(1.0, rmin / r)[:, None]
        P[:, [0, 2]] *= s
        P[:, 1] = y
        return P

    @staticmethod
    def _slide(P, frac, half=1.6, n=6):
        """Sub-segment of polyline P centred at fraction frac, interpolated (no point-to-point jumps)."""
        pos = frac * (len(P) + 2 * half) - half
        idx = np.clip(np.linspace(pos - half, pos + half, n), 0, len(P) - 1)
        if idx[-1] - idx[0] < 0.2:
            return P[:0]
        xs = np.arange(len(P))
        return np.column_stack([np.interp(idx, xs, P[:, j]) for j in range(3)])

    def _draw_chip(self, state, t, prim, sec, k):
        """Hexagonal implant on the forehead with two circuit traces running to the temple."""
        u, fem = self.u, self.fem
        hot = (255, 255, 255)
        cy, cp = 0.43, 0.0
        a = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
        for r, kk, w in ((1.0, 0.95, 1.8), (0.55, 0.6, 1.3)):
            P = self._surface(cy + 0.055 * r * np.sin(a), cp + 0.075 * r * np.cos(a), 0.0, fem) * 1.02
            self._glow_poly(P, prim, k * kk, w * u, layers=((5, 0.12), (2.4, 0.35), (1, 1.0)))
        rate = {"thinking": 6.0, "speaking": 3.0, "listening": 1.5}.get(state, 0.8)
        pulse = 0.5 + 0.5 * math.sin(t * rate)
        if state == "speaking":
            pulse = max(pulse * 0.5, self.jaw)
        core = self._surface(np.array([cy, cy]), np.array([cp - 0.012, cp + 0.012]), 0.0, fem) * 1.025
        self._glow_poly(core, lerp_rgb(prim, hot, 0.6), k * (0.4 + 0.6 * pulse), 4 * u,
                        layers=((4, 0.2), (2, 0.5), (1, 1.0)))
        for path in (((cy + 0.01, cp + 0.075), (cy + 0.01, 0.30), (0.52, 0.42), (0.52, 0.62)),
                     ((cy - 0.02, cp + 0.07), (cy - 0.02, 0.26), (0.36, 0.36), (0.36, 0.58))):
            d = self._densify(list(path), 6)
            P = self._surface(d[:, 0], d[:, 1], 0.0, fem) * 1.015
            self._trace(P, prim, k * 0.65, 1.2 * u, layers=((3, 0.15), (1, 1.0)))
            seg = self._slide(P, (t * rate * 0.15) % 1)
            if len(seg) >= 2 and state != "idle":
                self._trace(seg, hot, k * 0.7, 1.8 * u, layers=((3, 0.3), (1, 1.0)))
            end = P[-1:]
            if self._visible(end)[0]:
                px, py, _ = self._project(end)
                r = 3 * u
                self.c.create_oval(px[0] - r, py[0] - r, px[0] + r, py[0] + r, outline=hexcol(prim, k),
                                   width=max(1, 1.3 * u), fill=hexcol((16, 13, 20), 1), tags="dyn")

    def _draw_undercut(self, state, t, prim, sec, k):
        """Neon stripes shaved into the short side, with a pulse running along them."""
        u, fem = self.u, self.fem
        hot = (255, 255, 255)
        speed = {"thinking": 0.9, "speaking": 0.6, "listening": 0.3}.get(state, 0.0)
        for i, y0 in enumerate((0.36, 0.27, 0.18)):
            ph = np.linspace(1.0 + 0.08 * i, 2.15 - 0.1 * i, 24)
            P = self._surface(y0 - 0.04 * (ph - 1.0), ph, 0.0, fem) * 1.012
            self._trace(P, prim, k * 0.7, 1.4 * u, layers=((4, 0.1), (2, 0.3), (1, 1.0)))
            if speed:
                seg = self._slide(P, (t * speed + i * 0.3) % 1)
                if len(seg) >= 2:
                    self._trace(seg, hot, k * 0.7, 2 * u, layers=((3, 0.3), (1, 1.0)))

    def _draw_hair(self, state, t, prim, sec, k):
        u, fem = self.u, self.fem
        if not hasattr(self, "hair_paths"):
            rng = random.Random(11)
            self.hair_paths = []
            for side in (-1, 1):
                for i, ph in enumerate(np.linspace(1.0, 2.6, 5)):
                    end = -1.15 - 0.35 * rng.random() - (0.1 if i < 3 else 0)
                    y = np.linspace(1.0, end, 46)
                    self.hair_paths.append((y, np.full_like(y, side * ph), rng.random() * 6))
            for d in (0.0, 0.07, 0.14):   # side-swept fringe across the forehead
                pts = [(0.99, 0.3 + d), (0.8, -0.05 + d), (0.6, -0.55 + d * 0.6), (0.42, -0.95 + d * 0.3),
                       (0.2, -1.2), (-0.3, -1.3 + d), (-1.25 - d, -1.35 + d)]
                arr = self._densify(pts, 7)
                self.hair_paths.append((arr[:, 0], arr[:, 1], 1.7 + d * 10))
        glow = (4, 0.1), (2, 0.3), (1, 1.0)
        for y, phi, off in self.hair_paths:
            P = self._hair_points(y, phi, t, fem, off, 1.1)
            col = lerp_rgb(sec, prim, 0.5 + 0.5 * math.sin(off))
            self._trace(P, col, k * 0.45, 1.2 * u, layers=glow)
        # a shimmer running down the hair while speaking
        if state == "speaking":
            y, phi, off = self.hair_paths[int(t * 3) % len(self.hair_paths)]
            n = len(y)
            pos = int(((t * 0.8) % 1) * n)
            seg = slice(max(0, pos - 3), min(n, pos + 3))
            P = self._hair_points(y, phi, t, fem, off, 1.1)[seg]
            if len(P) >= 2:
                self._trace(P, (255, 255, 255), k * 0.6, 1.8 * u, layers=((3, 0.3), (1, 1.0)))

    # ---- live weather: neon rain, snow on the grid, lightning ---------------------------
    def _ground_y(self, s):
        """s: 0 at the horizon .. 1 at the bottom edge, same curve as the grid rows."""
        return self.horizon + (self.H - self.horizon) * s

    def _lane_x(self, lane, s):
        """x of a point on the floor that rides the grid's vanishing lines."""
        return self.W / 2 + lane * self.W * (0.02 + 0.14 * s)

    def _build_weather(self):
        c, W, H, u = self.c, self.W, self.H, self.u
        rng = self.rng = np.random.default_rng()
        self.horizon = H * 0.72
        self.rain_lvl = self.snow_lvl = self.flash = 0.0
        self.slant, self.wx_last, self.wx_cols, self.cur_bg = 0.0, None, {}, None
        self.next_bolt, self.bolt = time.monotonic() + 3, None
        # rain: x, y of the drop's head, depth 0 (far) .. 1 (near)
        self.rd = np.sort(rng.uniform(0, 1, N_RAIN) ** 1.4)
        self.rx = rng.uniform(-0.2 * W, 1.2 * W, N_RAIN)
        self.ry = np.zeros(N_RAIN)
        self.r_on = np.zeros(N_RAIN, bool)
        lay = np.minimum(3, (self.rd * 4).astype(int))
        self.r_glow = [c.create_line(0, 0, 0, 0, width=max(1, u * (1.5 + 3.5 * d)), capstyle="round",
                                     state="hidden", tags=("wx", f"wxg{l}")) for d, l in zip(self.rd, lay)]
        self.r_core = [c.create_line(0, 0, 0, 0, width=max(1, u * (0.6 + 1.6 * d)), capstyle="round",
                                     state="hidden", tags=("wx", f"wxc{l}")) for d, l in zip(self.rd, lay)]
        # splash rings on the floor
        self.sp = np.full((N_SPLASH, 4), -1.0)    # x, y, r_max, age (-1 = free)
        self.sp_items = [c.create_oval(0, 0, 0, 0, outline="", width=max(1, 1.5 * u), state="hidden",
                                       tags="wx") for _ in range(N_SPLASH)]
        # snow: x, y, depth, sway phase; settled flakes ride the floor (lane, s, age)
        self.fd = np.sort(rng.uniform(0, 1, N_SNOW) ** 1.3)
        self.fx = rng.uniform(-0.1 * W, 1.1 * W, N_SNOW)
        self.fy = np.zeros(N_SNOW)
        self.fph = rng.uniform(0, 2 * math.pi, N_SNOW)
        self.f_mode = np.zeros(N_SNOW, int)       # 0 hidden, 1 falling, 2 settled
        self.f_lane, self.f_s, self.f_age = np.zeros(N_SNOW), np.zeros(N_SNOW), np.zeros(N_SNOW)
        self.f_q = np.full(N_SNOW, -1)
        flay = np.minimum(3, (self.fd * 4).astype(int))
        self.f_glow = [c.create_oval(0, 0, 0, 0, outline="", state="hidden", tags=("wx", f"wsg{l}"))
                       for l in flay]
        self.f_core = [c.create_polygon(0, 0, 0, 0, 0, 0, outline="", state="hidden", tags=("wx", f"wsc{l}"))
                       for l in flay]
        self.f_lay = flay

    def _draw_weather(self, t, prim, sec, k):
        wx, W, H, u, hz = self.weather, self.W, self.H, self.u, self.horizon
        now = time.monotonic()
        dt = min(0.1, now - (self.wx_last or now))
        self.wx_last = now
        rng, w, out = self.rng, self.c._w, []
        ease = min(1.0, dt * 0.6)                 # ~2-3 s fade when the weather changes
        self.rain_lvl = lerp(self.rain_lvl, wx.level if wx.kind in ("rain", "storm") else 0.0, ease)
        self.snow_lvl = lerp(self.snow_lvl, wx.level if wx.kind == "snow" else 0.0, ease)
        self.slant = lerp(self.slant, wx.wind_x * 0.35, ease)
        if self.rain_lvl < 0.005 and self.snow_lvl < 0.005 and not self.r_on.any() \
                and not self.f_mode.any() and self.flash < 0.01 and (self.sp[:, 3] < 0).all():
            return
        kk = max(k, 0.6) * (1 - 0.3 * getattr(self, "presence", 0.0))   # step back when the face is up

        # colours per depth layer, only re-sent when they change
        core = lerp_rgb(sec, (235, 240, 255), 0.4)
        snow = lerp_rgb((245, 240, 255), sec, 0.15)
        for l in range(4):
            b = 0.3 + 0.7 * (l + 0.5) / 4
            for tag, col in ((f"wxc{l}", hexcol(core, kk * b * 0.9)), (f"wxg{l}", hexcol(sec, kk * b * 0.28)),
                             (f"wsc{l}", hexcol(snow, min(1.0, kk * b * 1.3))), (f"wsg{l}", hexcol(sec, kk * b * 0.2))):
                if self.wx_cols.get(tag) != col:
                    out.append(f"{w} itemconfigure {tag} -fill {col}")
                    self.wx_cols[tag] = col

        # ---- rain
        idx = np.arange(N_RAIN)
        active = idx < int(round(self.rain_lvl * N_RAIN))
        was_on = self.r_on.copy()
        on = self.r_on
        step = H * (1.0 + 1.7 * self.rd) * dt
        self.ry[on] += step[on]
        self.rx[on] += step[on] * self.slant
        length = H * (0.02 + 0.065 * self.rd)
        gs = self.rd ** 1.5
        gy = hz + (H - hz) * gs
        landed = on & (self.ry >= gy)
        for i in np.flatnonzero(landed & (self.rd > 0.3)):
            if rng.random() < 0.55:
                free = np.flatnonzero(self.sp[:, 3] < 0)
                if len(free):
                    self.sp[free[0]] = (self.rx[i], gy[i], u * (8 + 30 * self.rd[i]), 0.0)
        respawn = active & (landed | ~on)
        n = int(respawn.sum())
        if n:
            first = ~was_on & respawn & (self.rain_lvl < wx.level * 0.9)   # first wave: spread out
            self.ry[respawn] = -rng.uniform(0, H * 0.5, n) - length[respawn]
            self.ry[first] = rng.uniform(-H * 0.6, H * 0.3, int(first.sum()))
            self.rx[respawn] = rng.uniform(-0.2 * W, 1.2 * W, n) - self.slant * H * 0.6
        on[respawn] = True
        on[landed & ~active] = False
        for i in np.flatnonzero(on):
            y1 = min(self.ry[i], gy[i])
            y0 = self.ry[i] - length[i]
            if y0 >= y1:
                continue
            x1 = self.rx[i] - (self.ry[i] - y1) * self.slant
            x0 = self.rx[i] - length[i] * self.slant
            pts = f"{x0:.0f} {y0:.0f} {x1:.0f} {y1:.0f}"
            out.append(f"{w} coords {self.r_glow[i]} {pts}")
            out.append(f"{w} coords {self.r_core[i]} {pts}")
        for i in np.flatnonzero(on != was_on):
            st = "normal" if on[i] else "hidden"
            out.append(f"{w} itemconfigure {self.r_glow[i]} -state {st}")
            out.append(f"{w} itemconfigure {self.r_core[i]} -state {st}")

        # ---- splash rings: expand and fade, flattened by perspective
        for j in np.flatnonzero(self.sp[:, 3] >= 0):
            x, y, rmax, age = self.sp[j]
            age += dt
            life = 0.4
            if age >= life:
                self.sp[j, 3] = -1
                out.append(f"{w} itemconfigure {self.sp_items[j]} -state hidden")
                continue
            self.sp[j, 3] = age
            f = age / life
            r = rmax * (0.25 + 0.75 * f ** 0.6)
            col = hexcol(core, kk * 0.8 * (1 - f) ** 1.5)
            out.append(f"{w} coords {self.sp_items[j]} {x - r:.0f} {y - r * 0.22:.0f} {x + r:.0f} {y + r * 0.22:.0f}")
            out.append(f"{w} itemconfigure {self.sp_items[j]} -state normal -outline {col}")

        # ---- snow
        n_act = int(round(self.snow_lvl * N_SNOW))
        for i in range(N_SNOW):
            mode, d = self.f_mode[i], self.fd[i]
            if mode == 0:
                if i >= n_act:
                    continue
                self.fx[i] = rng.uniform(-0.1 * W, 1.1 * W)
                self.fy[i] = rng.uniform(-H * 0.7, 0) if self.snow_lvl > wx.level * 0.9 else rng.uniform(-H * 0.2, H * 0.6)
                self.f_mode[i] = mode = 1
                out.append(f"{w} itemconfigure {self.f_glow[i]} -state normal")
                out.append(f"{w} itemconfigure {self.f_core[i]} -state normal")
            size = u * (3.2 + 8 * d)
            if mode == 1:
                self.fy[i] += H * (0.04 + 0.1 * d) * dt
                self.fx[i] += (math.sin(t * 1.1 + self.fph[i]) * 25 + self.slant * 120) * u * (0.4 + d) * dt
                s_g = d ** 1.7
                if self.fy[i] >= self._ground_y(s_g):
                    self.f_mode[i] = mode = 2
                    self.f_s[i], self.f_age[i] = s_g, 0.0
                    self.f_lane[i] = (self.fx[i] - W / 2) / (W * (0.02 + 0.14 * s_g))
                x, y = self.fx[i], self.fy[i]
            if mode == 2:
                # settle on the floor and ride it toward the viewer, fading out
                f = self.f_s[i] ** (1 / 2.2) + 0.6 / 14 * dt
                self.f_s[i] = s = f ** 2.2
                self.f_age[i] += dt
                x, y = self._lane_x(self.f_lane[i], s), self._ground_y(s)
                size *= 0.8 + 0.6 * s
                fade = 1 - self.f_age[i] / 5.0
                if fade <= 0 or y > H + 10:
                    self.f_mode[i] = 0
                    self.f_q[i] = -1
                    out.append(f"{w} itemconfigure {self.f_glow[i]} -state hidden")
                    out.append(f"{w} itemconfigure {self.f_core[i]} -state hidden")
                    out.append(f"{w} addtag wsc{self.f_lay[i]} withtag {self.f_core[i]}")
                    out.append(f"{w} itemconfigure {self.f_core[i]} "
                               f"-fill {self.wx_cols.get(f'wsc{self.f_lay[i]}', '#ffffff')}")
                    continue
                q = int(fade * 6)
                if q != self.f_q[i]:
                    self.f_q[i] = q
                    b = (0.3 + 0.7 * (self.f_lay[i] + 0.5) / 4) * (q + 1) / 7
                    # take the flake out of its layer tag so the layer colour doesn't override it
                    out.append(f"{w} dtag {self.f_core[i]} wsc{self.f_lay[i]}")
                    out.append(f"{w} itemconfigure {self.f_core[i]} -fill {hexcol(snow, kk * b)}")
                    out.append(f"{w} itemconfigure {self.f_glow[i]} -state hidden")
            sw = size * 0.7
            out.append(f"{w} coords {self.f_core[i]} {x:.0f} {y - size:.0f} {x + sw:.0f} {y:.0f} "
                       f"{x:.0f} {y + size:.0f} {x - sw:.0f} {y:.0f}")
            if mode == 1:
                g = size * 1.7
                out.append(f"{w} coords {self.f_glow[i]} {x - g:.0f} {y - g:.0f} {x + g:.0f} {y + g:.0f}")

        if out:
            self.c.tk.eval("\n".join(out))

        # ---- lightning: a jagged bolt behind the horizon, the sky and floor flash
        if wx.kind == "storm" and self.rain_lvl > 0.3 and now > self.next_bolt:
            self.next_bolt = now + random.uniform(5, 15)
            x = random.uniform(0.1 * W, 0.9 * W)
            pts, y = [(x, -10)], -10
            while y < hz:
                y = min(hz, y + random.uniform(0.04, 0.09) * H)
                x += random.uniform(-0.045, 0.045) * W
                pts.append((x, y))
            fork = pts[random.randint(2, max(2, len(pts) // 2))]
            branch, (bx, by) = [fork], fork
            for _ in range(random.randint(2, 4)):
                bx += random.choice((-1, 1)) * random.uniform(0.02, 0.05) * W
                by += random.uniform(0.04, 0.08) * H
                branch.append((bx, min(by, hz)))
            self.bolt = (now, pts, branch)
        fl = 0.0
        if self.bolt:
            age = now - self.bolt[0]
            if age > 0.5:
                self.bolt = None
            else:
                fl = 1.0 if age < 0.07 else 0.25 if age < 0.13 else 0.9 if age < 0.2 else (0.5 - age) / 0.3 * 0.6
                for path, width in ((self.bolt[1], 2.4), (self.bolt[2], 1.3)):
                    flat = [v for p in path for v in p]
                    for w_mul, b_mul in ((7, 0.12), (3, 0.35), (1, 1.0)):
                        col = hexcol(lerp_rgb(sec, (255, 255, 255), 0.6 if w_mul == 1 else 0), fl * b_mul)
                        self.c.create_line(*flat, fill=col, width=max(1, width * w_mul * u),
                                           capstyle="round", joinstyle="round", tags=("dyn", "bolt"))
                self.c.tag_raise("bolt", "sky")   # just above the (opaque) sky, below grid and head
        self.flash = max(fl, self.flash * 0.75)
        if self.flash < 0.01:
            self.flash = 0.0
        bg = hexcol(lerp_rgb(BG, (60, 14, 34), self.flash * 0.6))
        if bg != self.cur_bg:
            self.cur_bg = bg
            self.c.configure(bg=bg)

    def _draw_hud(self, state, t, prim, sec, k):
        dots = "." * (int(t * 3) % 4) if state in ("thinking", "listening") else ""
        flicker = 0.85 + 0.15 * random.random()
        self.c.itemconfigure(self.label, text=f"// {LABELS[state]}{dots}", fill=hexcol(prim, k * flicker))
        self.c.itemconfigure(self.sub, text=time.strftime("HEY CLAUDE  //  %H:%M:%S  //  VOICE LINK ")
                             + ("ONLINE" if state != "idle" else "STANDBY")
                             + ("  //  FR" if state != "idle" and self.fem > 0.5 else "")
                             + self.weather.hud(), fill=hexcol(sec, 0.6 * k))
        self.c.tag_raise("hud")
        self.c.tag_raise("scan")


def pin_cpus():
    """Optional CPU pinning, e.g. FACE_CPUS=8-15 for the E-cores of this Core Ultra 5 125H.
    Off by default: on E-cores frames took ~40 of 50 ms and dropped to ~10 fps whenever
    Whisper/Kokoro were busy. Empty disables pinning; "8-15" or "8,9,10" style lists are accepted."""
    spec = os.environ.get("FACE_CPUS", "").strip()
    if not spec:
        return
    cpus = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    try:
        os.sched_setaffinity(0, cpus & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    except (OSError, ValueError):
        pass


def main():
    pin_cpus()
    face = Face()
    try:
        face.root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
