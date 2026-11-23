"""Living sky for the face wallpaper: real sun / moon / stars, northern lights, ISS flyovers,
meteor showers, and the computer's "vital signs".

Everything is computed for the location in ~/.config/hey-claude/weather.conf (never shown).
Data: NOAA SWPC (Kp index), CelesTrak (ISS orbit, cached for a day), Open-Meteo (via Weather).

Testing overrides (environment):
  FACE_SKY_TIME=2026-11-16T22:30   pretend it is this local time (the clock keeps running)
  FACE_AURORA=6                    force a Kp index
  FACE_METEORS=20                  force meteors per minute
  FACE_ISS=1                       fake an ISS pass right away (repeats every 3 min)
  FACE_CLOUD=0..100                force cloud cover
  FACE_VITALS=cpu=90,temp=95,bat=8 force CPU %, CPU temperature, battery % (discharging)
"""
import datetime
import json
import math
import os
import random
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "hey-claude"
RAD, DEG = math.pi / 180, 180 / math.pi


def _env_float(name):
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


_SKY_TIME = os.environ.get("FACE_SKY_TIME")
TIME_OFFSET = 0.0
if _SKY_TIME:
    try:
        TIME_OFFSET = datetime.datetime.fromisoformat(_SKY_TIME).timestamp() - time.time()
    except ValueError:
        pass


MOON_ELONG = None  # showcase: force the moon's phase angle (radians, 0 = new, pi = full)
COLOUR_DT = 1.0   # seconds between sky gradient recolours (the showcase lowers it)
SLOW_DT = 2.0     # seconds between sun/moon position updates (the showcase lowers it)


def sky_now():
    return time.time() + TIME_OFFSET


TEST_MODE = any(os.environ.get(v) for v in (
    "FACE_SKY_TIME", "FACE_AURORA", "FACE_METEORS", "FACE_ISS", "FACE_CLOUD", "FACE_VITALS"))


def notify(title, body, icon="weather-clear-night"):
    if TEST_MODE:
        print(f"notify (suppressed in test mode): {title} / {body}", flush=True)
        return
    try:
        subprocess.Popen(["notify-send", "-a", "Jarvis sky", "-i", icon, title, body],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
    print(f"notify: {title} / {body}", flush=True)


def fetch_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "desktop-wallpaper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---- low-precision astronomy (good to ~1 degree, plenty for a wallpaper) --------------
def days_j2000(ts):
    return ts / 86400.0 + 2440587.5 - 2451545.0


def gmst_deg(d):
    return (280.46061837 + 360.98564736629 * d) % 360


def sun_radec(d):
    g = (357.529 + 0.98560028 * d) * RAD
    q = 280.459 + 0.98564736 * d
    lam = (q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) * RAD
    eps = (23.439 - 0.00000036 * d) * RAD
    return math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam)), math.asin(math.sin(eps) * math.sin(lam)), lam


def moon_radec(d):
    L = (218.316 + 13.176396 * d) * RAD
    M = (134.963 + 13.064993 * d) * RAD
    F = (93.272 + 13.229350 * d) * RAD
    lam = L + 6.289 * RAD * math.sin(M)
    beta = 5.128 * RAD * math.sin(F)
    eps = 23.439 * RAD
    ra = math.atan2(math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps), math.cos(lam))
    dec = math.asin(math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam))
    return ra, dec, lam


def alt_az(ra, dec, d, lat, lon):
    """-> altitude, azimuth (degrees, azimuth from north through east)."""
    H = (gmst_deg(d) + lon) * RAD - ra
    phi = lat * RAD
    alt = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(H))
    az = math.atan2(-math.cos(dec) * math.sin(H),
                    math.sin(dec) * math.cos(phi) - math.cos(dec) * math.cos(H) * math.sin(phi))
    return alt * DEG, (az * DEG) % 360


COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def compass(az):
    return COMPASS[int((az + 22.5) % 360 // 45)]


# ---- meteor showers: (name, month, day of peak, ZHR, half-width in days) ---------------
SHOWERS = [
    ("Quadrantids", 1, 3, 110, 0.6), ("Lyrids", 4, 22, 18, 1.2), ("Eta Aquariids", 5, 6, 30, 2.5),
    ("Delta Aquariids", 7, 30, 20, 3.0), ("Perseids", 8, 12, 100, 2.0), ("Draconids", 10, 8, 10, 0.7),
    ("Orionids", 10, 21, 20, 2.5), ("Leonids", 11, 17, 15, 1.2), ("Geminids", 12, 14, 150, 1.3),
    ("Ursids", 12, 22, 10, 0.8),
]


def meteor_rate(ts):
    """Meteors per minute to show (sporadics + active showers), and the active shower name."""
    forced = _env_float("FACE_METEORS")
    if forced is not None:
        return forced, "TEST"
    now = datetime.datetime.fromtimestamp(ts)
    best, name = 0.0, ""
    for nm, m, dd, zhr, width in SHOWERS:
        for year in (now.year - 1, now.year, now.year + 1):
            days = (now - datetime.datetime(year, m, dd)).total_seconds() / 86400
            act = zhr * math.exp(-0.5 * (days / width) ** 2)
            if act > best:
                best, name = act, nm
    return 0.25 + best / 25, (name if best > 3 else "")


# ---- background data ------------------------------------------------------------------
class SpaceData:
    """Kp index (aurora) and the ISS orbit."""

    def __init__(self):
        self.kp = _env_float("FACE_AURORA") or 0.0
        self.sat = None
        self.fake_iss = os.environ.get("FACE_ISS") == "1"
        self.fake_t0 = time.monotonic()
        self.fake_cross = 90.0         # seconds a fake pass takes (then the same time gone)
        if _env_float("FACE_AURORA") is None:
            threading.Thread(target=self._kp_loop, daemon=True).start()
        threading.Thread(target=self._tle_loop, daemon=True).start()

    def _kp_loop(self):
        ok_t = time.time()
        while True:
            try:
                rows = fetch_json("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
                recent = [float(r.get("estimated_kp") or r.get("kp_index") or 0) for r in rows[-15:]]
                self.kp = max(recent) if recent else 0.0
                ok_t = time.time()
                print(f"space weather: Kp {self.kp:.2f}", flush=True)
                time.sleep(600)
            except Exception as e:
                print("Kp fetch failed:", e, flush=True)
                if time.time() - ok_t > 3 * 3600:     # stale: don't keep showing/notifying an old storm
                    self.kp = 0.0
                time.sleep(180)

    def _tle_loop(self):
        from sgp4.api import Satrec
        path = CACHE / "iss.tle"
        while True:
            try:
                if not path.exists() or time.time() - path.stat().st_mtime > 86400:
                    try:
                        req = urllib.request.Request(
                            "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE",
                            headers={"User-Agent": "desktop-wallpaper/1.0"})
                        with urllib.request.urlopen(req, timeout=20) as r:
                            text = r.read().decode()
                        if "1 25544" in text:
                            CACHE.mkdir(parents=True, exist_ok=True)
                            path.write_text(text)
                    except Exception as e:      # offline: fall back to the cached (older) orbit
                        print("ISS orbit fetch failed, using cache:", e, flush=True)
                lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
                l1 = next(l for l in lines if l.startswith("1 "))
                l2 = next(l for l in lines if l.startswith("2 "))
                self.sat = Satrec.twoline2rv(l1, l2)
                time.sleep(3600)
            except Exception as e:
                print("ISS orbit fetch failed:", e, flush=True)
                time.sleep(600)

    def iss(self, ts, lat, lon):
        """-> (alt, az, sunlit) of the ISS, or None."""
        if self.fake_iss:
            fc = self.fake_cross
            f = ((time.monotonic() - self.fake_t0) % (2 * fc)) / fc  # fc s across, fc s gone
            if f > 1:
                return -10.0, 0.0, True
            return 65 * math.sin(math.pi * f) - 2, (230 + 160 * f) % 360, True
        sat = self.sat
        if sat is None:
            return None
        jd = ts / 86400.0 + 2440587.5
        err, r, _ = sat.sgp4(math.floor(jd), jd - math.floor(jd))
        if err:
            return None
        d = days_j2000(ts)
        th = (gmst_deg(d) + lon) * RAD
        phi = lat * RAD
        R = 6371.0
        obs = (R * math.cos(phi) * math.cos(th), R * math.cos(phi) * math.sin(th), R * math.sin(phi))
        rx, ry, rz = r[0] - obs[0], r[1] - obs[1], r[2] - obs[2]
        S = math.sin(phi) * math.cos(th) * rx + math.sin(phi) * math.sin(th) * ry - math.cos(phi) * rz
        E = -math.sin(th) * rx + math.cos(th) * ry
        Z = math.cos(phi) * math.cos(th) * rx + math.cos(phi) * math.sin(th) * ry + math.sin(phi) * rz
        rng = math.sqrt(rx * rx + ry * ry + rz * rz)
        alt = math.asin(Z / rng) * DEG
        az = math.atan2(E, -S) * DEG % 360
        # in Earth's shadow? (cylindrical shadow model)
        ra, dec, _ = sun_radec(d)
        s = (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))
        proj = r[0] * s[0] + r[1] * s[1] + r[2] * s[2]
        perp = math.sqrt(max(0.0, r[0] ** 2 + r[1] ** 2 + r[2] ** 2 - proj * proj))
        sunlit = proj > 0 or perp > 6371.0
        return alt, az, sunlit


class Vitals:
    """CPU load, CPU package temperature and battery, polled once a second."""

    def __init__(self):
        self.cpu, self.temp, self.bat, self.discharging = 0.0, 50.0, 100.0, False
        forced = os.environ.get("FACE_VITALS")
        self.forced = {}
        if forced:
            for part in forced.split(","):
                k, _, v = part.partition("=")
                try:
                    self.forced[k.strip()] = float(v)
                except ValueError:
                    pass
        self._temp_path = None
        for hw in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                if (hw / "name").read_text().strip() in ("coretemp", "k10temp", "zenpower"):
                    self._temp_path = hw / "temp1_input"
            except OSError:
                pass
        self._bat = next(iter(sorted(Path("/sys/class/power_supply").glob("BAT*"))), None)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = None
        while True:
            try:
                f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
                idle, total = f[3] + f[4], sum(f)
                if prev:
                    dt = total - prev[1]
                    busy = 1 - (idle - prev[0]) / dt if dt else 0
                    self.cpu = self.cpu * 0.5 + max(0.0, min(1.0, busy)) * 0.5
                prev = (idle, total)
                if self._temp_path:
                    self.temp = int(self._temp_path.read_text()) / 1000
                if self._bat:
                    self.bat = float((self._bat / "capacity").read_text())
                    self.discharging = (self._bat / "status").read_text().strip() == "Discharging"
            except (OSError, ValueError):
                pass
            if "cpu" in self.forced:
                self.cpu = self.forced["cpu"] / 100
            if "temp" in self.forced:
                self.temp = self.forced["temp"]
            if "bat" in self.forced:
                self.bat, self.discharging = self.forced["bat"], True
            time.sleep(1)

    def grid_speed(self):
        return 1.0 + 2.2 * self.cpu ** 1.5

    def heat(self):
        return max(0.0, min(1.0, (self.temp - 68) / 24))

    def power_k(self):
        """Scene brightness on a low battery: dims below 15 %, and flickers more as it drains."""
        if not self.discharging or self.bat > 15:
            return 1.0
        low = 1 - self.bat / 15
        k = 1.0 - 0.45 * low
        if random.random() < 0.004 + 0.05 * low ** 2:
            k *= random.uniform(0.15, 0.5)
        return k


# ---- drawing ----------------------------------------------------------------------------
SKY_KEYS = [  # sun altitude -> (top colour, horizon colour)
    (-90, (3, 1, 6), (12, 3, 18)),
    (-18, (3, 1, 6), (12, 3, 18)),
    (-10, (5, 2, 14), (34, 8, 46)),
    (-4, (10, 3, 26), (105, 22, 70)),
    (1, (18, 6, 38), (160, 45, 60)),
    (8, (22, 8, 44), (110, 30, 70)),
    (90, (20, 8, 42), (64, 20, 66)),
]


def _mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, c))) for c in rgb)


def sky_palette(sun_alt):
    for (a0, t0, h0), (a1, t1, h1) in zip(SKY_KEYS, SKY_KEYS[1:]):
        if a0 <= sun_alt <= a1:
            f = (sun_alt - a0) / (a1 - a0)
            return _mix(t0, t1, f), _mix(h0, h1, f)
    return SKY_KEYS[-1][1], SKY_KEYS[-1][2]


class Sky:
    N_BANDS, N_STARS, N_METEORS = 26, 170, 4
    RAYS = (96, 64)                      # rays per aurora curtain

    def __init__(self, face, location):
        self.f = face
        self.c = face.c
        self.W, self.H, self.u = face.W, face.H, face.u
        self.hz = face.H * 0.72
        self.location = location            # callable -> (lat, lon) or None
        self.space = SpaceData()
        self.vitals = face.vitals
        self.rng = np.random.default_rng()
        self.cols = {}
        self.slow_t = -99.0
        self.sun_alt, self.sun_az, self.moon = -30.0, 180.0, None
        self.star_vis = 0.0
        self.aurora_k = 0.0
        self.meteors = []                   # dicts
        self.next_meteor = time.monotonic() + 5
        self.iss_state = None
        self.iss_trail = []
        self.iss_notified = False
        self.aurora_notified = 0.0
        self.last = None

    # -- helpers
    def sky_xy(self, alt, az):
        """Panorama: south in the middle, east left, west right, north at both edges."""
        a = ((az - 180 + 180) % 360) - 180
        x = self.W / 2 + a / 180 * self.W / 2
        y = self.hz - (max(alt, -10) / 90) * self.hz * 0.95
        return x, y

    def sky_rgb(self, y):
        """Current sky colour at height y (what a glow should blend into)."""
        bands = getattr(self, "band_rgb", None)
        if not bands:
            return (3, 1, 6)
        return bands[max(0, min(len(bands) - 1, int(y / self.hz * len(bands))))]

    def circle(self, x, y, r, n=32):
        """Polygon points of a circle, flattened at the horizon so nothing spills onto the floor."""
        return " ".join(f"{x + r * math.cos(v):.0f} {min(self.hz, y + r * math.sin(v)):.0f}"
                        for v in np.linspace(0, 2 * math.pi, n, endpoint=False))

    def cloud(self):
        forced = _env_float("FACE_CLOUD")
        if forced is not None:
            return forced / 100
        return (getattr(self.f.weather, "cloud", None) or 0) / 100

    def _set(self, out, tag, opt, col):
        key = (tag, opt)
        if self.cols.get(key) != col:
            self.cols[key] = col
            out.append(f"{self.c._w} itemconfigure {tag} -{opt} {col}")

    # -- build: called from Face._draw_static before the grid (so it sits behind)
    def build_back(self):
        c, W, H, u, hz = self.c, self.W, self.H, self.u, self.hz
        rng = self.rng
        # sky gradient bands (top -> horizon)
        self.bands = []
        for i in range(self.N_BANDS):
            y0, y1 = hz * i / self.N_BANDS, hz * (i + 1) / self.N_BANDS + 1
            self.bands.append(c.create_rectangle(0, y0, W, y1, outline="", fill="#030106", tags="sky"))
        # stars
        self.star_items = []
        self.star_base = rng.uniform(0.35, 1.0, self.N_STARS)
        for i in range(self.N_STARS):
            x, y = rng.uniform(0, W), hz * 0.93 * rng.uniform(0, 1) ** 1.25
            r = u * (0.9 + 1.6 * rng.uniform() ** 3)
            self.star_items.append(c.create_oval(x - r, y - r, x + r, y + r, outline="", fill="#030106",
                                                 state="hidden", tags=("sky", "star")))
        self.stars_shown = False
        # sun: glow + striped synthwave disc
        self.sun_glow = [c.create_polygon(0, 0, 0, 0, 0, 0, outline="", fill="#030106", state="hidden",
                                          tags=("sky", "sun")) for _ in range(3)]      # largest first
        self.sun_slices = [c.create_polygon(0, 0, 0, 0, 0, 0, outline="", fill="#030106", state="hidden",
                                            tags=("sky", "sun")) for _ in range(11)]
        # moon: glow, dark disc, lit part
        self.moon_glow = c.create_polygon(0, 0, 0, 0, 0, 0, outline="", fill="#030106", state="hidden",
                                          tags=("sky", "moon"))
        self.moon_disc = c.create_polygon(0, 0, 0, 0, 0, 0, outline="", fill="#030106", state="hidden",
                                          tags=("sky", "moon"))
        self.moon_lit = c.create_polygon(0, 0, 0, 0, 0, 0, outline="", fill="#030106", state="hidden",
                                         tags=("sky", "moon"))
        # aurora: 2 curtains of rays, each ray a lower (green) and upper (violet) segment,
        # drawn wide+dim (the curtain) and thin+bright (the ray)
        self.aur = []
        for band, n in enumerate(self.RAYS):
            items = []
            for i in range(n):
                lvl = (i * 7 + band) % 3                       # fixed bright/dim pattern along the curtain
                row = [c.create_line(0, 0, 0, 0, width=max(1, 7 * u), capstyle="round", fill="#030106",
                                     state="hidden", tags=("sky", "aur", f"aurg{band}{lvl}"))]
                for seg in range(3):
                    row.append(c.create_line(0, 0, 0, 0, width=max(1, (1.8 - 0.4 * seg) * u), capstyle="round",
                                             fill="#030106", state="hidden",
                                             tags=("sky", "aur", f"aur{band}{seg}{lvl}")))
                items.append(row)
            self.aur.append(items)
        self.aur_flip = 0
        self.aur_shown = False
        # meteors: 3 tail segments x (glow, core)
        self.met_items = [[c.create_line(0, 0, 0, 0, capstyle="round", fill="#030106", state="hidden",
                                         width=max(1, u * w), tags=("sky", "met"))
                           for _seg in range(3) for w in (5, 1.6)] for _ in range(self.N_METEORS)]
        # ISS: trail, glow, dot, label
        self.iss_trail_item = c.create_line(0, 0, 0, 0, fill="#030106", width=max(1, 1.2 * u), dash=(2, 4),
                                            state="hidden", tags=("sky", "iss"))
        self.iss_glow = c.create_oval(0, 0, 0, 0, outline="", fill="#030106", state="hidden", tags=("sky", "iss"))
        self.iss_dot = c.create_oval(0, 0, 0, 0, outline="", fill="#030106", state="hidden", tags=("sky", "iss"))
        self.iss_label = c.create_text(0, 0, text="ISS", anchor="w", font=("Monospace", int(9 * u), "bold"),
                                       fill="#030106", state="hidden", tags=("sky", "iss"))
        self.iss_shown = False

    # -- per frame
    def update(self, t, prim, sec, k, flash):
        now = time.monotonic()
        dt = min(0.1, now - (self.last or now))
        self.last = now
        out = []
        loc = self.location()
        ts = sky_now()
        cloud = self.cloud()
        pk = self.vitals.power_k()
        # the gradient only drifts with the sun; recolour it once a second unless lightning flashes
        if flash > 0.001 or pk < 1.0 or now - getattr(self, "colours_t", -9.0) > COLOUR_DT or getattr(self, "had_flash", False):
            self.colours_t = now
            self.had_flash = flash > 0.001
            self._sky_colours(cloud, flash, pk, out)
        if loc and now - self.slow_t > SLOW_DT:
            self.slow_t = now
            self._slow_update(ts, loc, cloud, out)
        self._stars(t, cloud, pk, out)
        self._aurora(t, ts, loc, cloud, pk, out)
        self._meteors(now, ts, cloud, pk, out)
        self._iss(ts, loc, cloud, pk, out)
        if out:
            self.c.tk.eval("\n".join(out))
        return pk

    def _slow_update(self, ts, loc, cloud, out):
        lat, lon = loc
        d = days_j2000(ts)
        ra, dec, sun_lam = sun_radec(d)
        self.sun_alt, self.sun_az = alt_az(ra, dec, d, lat, lon)
        mra, mdec, mlam = moon_radec(d)
        malt, maz = alt_az(mra, mdec, d, lat, lon)
        elong = (mlam - sun_lam) % (2 * math.pi)
        if MOON_ELONG is not None:                       # showcase: sweep the phase
            elong = MOON_ELONG % (2 * math.pi)
        self.moon = (malt, maz, elong)
        self._draw_sun(cloud, out)
        self._draw_moon(cloud, out)

    def _sky_colours(self, cloud, flash, pk, out):
        top, hor = sky_palette(self.sun_alt)
        grey = lambda c: (sum(c) / 3 * 0.75,) * 2 + (sum(c) / 3 * 0.9,)
        top, hor = _mix(top, grey(top), cloud * 0.6), _mix(hor, grey(hor), cloud * 0.6)
        fl = (60 * flash * 0.6, 14 * flash * 0.6, 34 * flash * 0.6)
        w = self.c._w
        self.band_rgb = []
        for i, item in enumerate(self.bands):
            f = (i / (self.N_BANDS - 1)) ** 2.2
            col = _mix(top, hor, f)
            col = tuple((col[j] + fl[j]) * pk for j in range(3))
            self.band_rgb.append(col)
            hx = _hex(col)
            if self.cols.get(item) != hx:
                self.cols[item] = hx
                out.append(f"{w} itemconfigure {item} -fill {hx}")

    def _draw_sun(self, cloud, out):
        w, H, hz = self.c._w, self.H, self.hz
        R = 0.085 * H
        x, cy = self.sky_xy(self.sun_alt, self.sun_az)
        visible = self.sun_alt > -6 and cy - R < hz
        if not visible:
            out.append(f"{w} itemconfigure sun -state hidden")
            return
        kk = 1 - 0.7 * cloud
        behind = self.sky_rgb(min(cy, hz - 1))
        for j, item in enumerate(self.sun_glow):          # j=0 is the outermost ring
            g = R * (2.25 - 0.45 * j)
            col = _hex(_mix(behind, (255, 90, 120), kk * (0.07 + 0.06 * j)))
            out.append(f"{w} coords {item} {self.circle(x, cy, g)}")
            out.append(f"{w} itemconfigure {item} -state normal -fill {col}")
        n = len(self.sun_slices)
        for j, item in enumerate(self.sun_slices):
            fa, fb = j / n, (j + 1) / n
            gap = max(0.0, (fa - 0.4) * 0.55) / n * 2 * R * 2.2      # stripes widen toward the bottom
            ya, yb = cy - R + fa * 2 * R + gap, cy - R + fb * 2 * R
            yb = min(yb, hz)
            if ya >= yb - 1:
                out.append(f"{w} itemconfigure {item} -state hidden")
                continue
            ys = np.linspace(ya, yb, 5)
            half = np.sqrt(np.maximum(0, R * R - (ys - cy) ** 2))
            pts = [f"{x - h:.0f} {y:.0f}" for y, h in zip(ys, half)] + \
                  [f"{x + h:.0f} {y:.0f}" for y, h in zip(ys[::-1], half[::-1])]
            col = _hex(_mix((3, 1, 6), _mix((255, 214, 90), (255, 30, 120), (fa + fb) / 2), kk))
            out.append(f"{w} coords {item} {' '.join(pts)}")
            out.append(f"{w} itemconfigure {item} -state normal -fill {col}")

    def _draw_moon(self, cloud, out):
        w = self.c._w
        if not self.moon or self.moon[0] < -3:
            out.append(f"{w} itemconfigure moon -state hidden")
            return
        malt, maz, elong = self.moon
        x, y = self.sky_xy(malt, maz)
        R = 0.032 * self.H
        night = max(0.35, min(1.0, (6 - self.sun_alt) / 12))
        kk = night * (1 - 0.75 * cloud)
        hz = self.hz
        behind = self.sky_rgb(min(y, hz - 1))
        dark = max(0.0, min(1.0, (-self.sun_alt - 2) / 10))     # the unlit part only shows at night
        out.append(f"{w} coords {self.moon_glow} {self.circle(x, y, R * 2.6, 28)}")
        out.append(f"{w} coords {self.moon_disc} {self.circle(x, y, R, 28)}")
        illum = (1 - math.cos(elong)) / 2
        glow_state = "normal" if dark > 0.05 else "hidden"
        out.append(f"{w} itemconfigure {self.moon_glow} -state {glow_state} -fill "
                   f"{_hex(_mix(behind, (120, 110, 190), kk * dark * 0.12 * (0.3 + illum)))}")
        out.append(f"{w} itemconfigure {self.moon_disc} -state {glow_state} -fill "
                   f"{_hex(_mix(behind, (70, 60, 110), kk * dark * 0.35))}")
        side = 1 if math.sin(elong) > 0 else -1          # waxing: lit on the right (north hemisphere)
        a = np.linspace(-math.pi / 2, math.pi / 2, 16)
        limb = [(x + side * R * math.cos(v), y + R * math.sin(v)) for v in a]
        term = [(x + side * R * math.cos(v) * math.cos(elong), y + R * math.sin(v)) for v in a[::-1]]
        pts = " ".join(f"{px:.0f} {min(hz, py):.0f}" for px, py in limb + term)
        out.append(f"{w} coords {self.moon_lit} {pts}")
        out.append(f"{w} itemconfigure {self.moon_lit} -state "
                   f"{'normal' if illum > 0.02 else 'hidden'} -fill {_hex(_mix(behind, (235, 230, 255), kk))}")

    def _stars(self, t, cloud, pk, out):
        dark = max(0.0, min(1.0, (-self.sun_alt - 5) / 10))
        vis = dark * (1 - cloud) ** 1.5 * pk
        self.star_vis = lerp(self.star_vis, vis, 0.05)
        w = self.c._w
        if self.star_vis < 0.02:
            if self.stars_shown:
                out.append(f"{w} itemconfigure star -state hidden")
                self.stars_shown = False
            return
        if not self.stars_shown:
            self.stars_shown = True
            for i, item in enumerate(self.star_items):
                col = _hex(_mix((3, 1, 6), (225, 215, 255), self.star_base[i] * self.star_vis))
                out.append(f"{w} itemconfigure {item} -state normal -fill {col}")
            return
        for i in random.sample(range(self.N_STARS), 14):          # twinkle a few per frame
            tw = random.uniform(0.45, 1.15)
            col = _hex(_mix((3, 1, 6), (225, 215, 255), min(1, self.star_base[i] * self.star_vis * tw)))
            out.append(f"{w} itemconfigure {self.star_items[i]} -fill {col}")

    def _aurora(self, t, ts, loc, cloud, pk, out):
        kp = self.space.kp
        dark = max(0.0, min(1.0, (-self.sun_alt - 8) / 6))
        target = max(0.0, min(1.0, (kp - 3.3) / 3.5)) * dark * (1 - 0.8 * cloud)
        self.aurora_k = lerp(self.aurora_k, target, 0.02)
        w = self.c._w
        # notification: strong enough to see, dark and not too cloudy
        if kp >= 5 and dark > 0.5 and cloud < 0.6 and time.time() - self.aurora_notified > 6 * 3600:
            self.aurora_notified = time.time()
            notify("Northern lights possible", f"Kp index {kp:.1f}. Go outside and look north!")
        if self.aurora_k < 0.02:
            if self.aur_shown:
                out.append(f"{w} itemconfigure aur -state hidden")
                self.aur_shown = False
            return
        if not self.aur_shown:
            out.append(f"{w} itemconfigure aur -state normal")
            self.aur_shown = True
        a = self.aurora_k * pk * (0.85 + 0.15 * math.sin(t * 0.7))
        seg_cols = ((110, 255, 170), (40, 200, 190), (170, 90, 255))    # green -> teal -> violet
        seg_k = (0.8, 0.45, 0.28)
        for band in range(2):
            ab = a * (1.0 if band == 0 else 0.65)
            for lvl in range(3):
                lk = (1.0, 0.6, 0.35)[lvl]
                self._set(out, f"aurg{band}{lvl}", "fill", _hex(_mix((3, 1, 6), (20, 170, 100), ab * lk * 0.16)))
                for seg in range(3):
                    self._set(out, f"aur{band}{seg}{lvl}", "fill",
                              _hex(_mix((3, 1, 6), seg_cols[seg], ab * lk * seg_k[seg])))
        # the curtains move slowly, so each frame only re-shapes half of the rays
        self.aur_flip ^= 1
        W, hz = self.W, self.hz
        for band, items in enumerate(self.aur):
            n_r = len(items)
            for i in range(self.aur_flip, n_r, 2):
                row = items[i]
                x = (i + 0.5) / n_r * W * 1.08 - 0.04 * W
                xx = x / W
                base = hz * (0.42 + 0.12 * band) + hz * 0.06 * math.sin(xx * 6.5 + t * 0.12 + band * 2.1) \
                    + hz * 0.025 * math.sin(xx * 17 - t * 0.3)
                n = 0.5 + 0.5 * math.sin(xx * 23 + t * 0.55 + band * 3) * math.sin(xx * 9 - t * 0.21 + band)
                h = hz * (0.08 + 0.22 * n) * (0.6 + 0.4 * self.aurora_k)
                lean = hz * 0.015 * math.sin(t * 0.4 + i * 0.25)        # rays sway
                ys = (base, base - h * 0.35, base - h * 0.7, base - h)
                xs = (x, x + lean * 0.35, x + lean * 0.7, x + lean)
                out.append(f"{w} coords {row[0]} {xs[0]:.0f} {ys[0]:.0f} {xs[1]:.0f} {ys[1]:.0f}")
                for seg in range(3):
                    out.append(f"{w} coords {row[seg + 1]} {xs[seg]:.0f} {ys[seg]:.0f} "
                               f"{xs[seg + 1]:.0f} {ys[seg + 1]:.0f}")

    def _meteors(self, now, ts, cloud, pk, out):
        w = self.c._w
        dark = max(0.0, min(1.0, (-self.sun_alt - 8) / 6))
        rate, _name = meteor_rate(ts)
        if dark > 0.3 and cloud < 0.85 and now > self.next_meteor and len(self.meteors) < self.N_METEORS:
            self.next_meteor = now + random.expovariate(max(rate, 0.1) / 60)
            free = [i for i in range(self.N_METEORS) if all(m["slot"] != i for m in self.meteors)]
            if free:
                ang = math.radians(random.uniform(25, 65)) * random.choice((-1, 1))
                self.meteors.append(dict(
                    slot=free[0], t0=now, dur=random.uniform(0.45, 0.95),
                    x=random.uniform(0.1, 0.9) * self.W, y=random.uniform(0.04, 0.45) * self.hz,
                    dx=math.sin(ang), dy=math.cos(ang), L=random.uniform(0.1, 0.28) * self.W,
                    k=random.uniform(0.6, 1.0) * dark * (1 - cloud) * pk))
        elif now > self.next_meteor:
            self.next_meteor = now + random.expovariate(max(rate, 0.1) / 60)
        for m in list(self.meteors):
            f = (now - m["t0"]) / m["dur"]
            items = self.met_items[m["slot"]]
            if f >= 1:
                for it in items:
                    out.append(f"{w} itemconfigure {it} -state hidden")
                self.meteors.remove(m)
                continue
            fade = min(1.0, (1 - f) / 0.3) * min(1.0, f / 0.1) * m["k"]
            hx, hy = m["x"] + m["dx"] * m["L"] * f, m["y"] + m["dy"] * m["L"] * f
            tail = m["L"] * 0.38 * min(1.0, f * 3)
            for s in range(3):
                a0, a1 = tail * s / 3, tail * (s + 1) / 3
                p = f"{hx - m['dx'] * a0:.0f} {hy - m['dy'] * a0:.0f} {hx - m['dx'] * a1:.0f} {hy - m['dy'] * a1:.0f}"
                b = fade * (1.0, 0.5, 0.2)[s]
                for j, (base, mul) in enumerate((((255, 60, 120), 0.25), ((255, 240, 250), 1.0))):
                    it = items[s * 2 + j]
                    out.append(f"{w} coords {it} {p}")
                    out.append(f"{w} itemconfigure {it} -state normal -fill {_hex(_mix((3, 1, 6), base, b * mul))}")

    def _iss(self, ts, loc, cloud, pk, out):
        w = self.c._w
        res = self.space.iss(ts, *loc) if loc else None
        if not res or res[0] < 1.5:  # hide at the horizon: nothing shows through the floor
            if self.iss_shown:
                out.append(f"{w} itemconfigure iss -state hidden")
                self.iss_shown = False
                self.iss_trail = []
            self.iss_notified = False
            return
        alt, az, sunlit = res
        x, y = self.sky_xy(alt, az)
        if self.iss_trail and abs(self.iss_trail[-1][0] - x) > self.W / 2:
            self.iss_trail = []          # crossed north (panorama edge): don't draw a line across the screen
        if not self.iss_trail or abs(self.iss_trail[-1][0] - x) + abs(self.iss_trail[-1][1] - y) > 6 * self.u:
            self.iss_trail.append((x, y))
            self.iss_trail = self.iss_trail[-60:]
        dark_sky = self.sun_alt < -4
        visible = sunlit and dark_sky
        if visible and alt > 10 and not self.iss_notified and cloud < 0.7:
            self.iss_notified = True
            notify("The ISS is overhead", f"Look {compass(az)}: a bright, steady moving dot.", "starred")
        bright = (1.0 if visible else 0.5) * pk
        core = (255, 255, 255) if sunlit else (255, 80, 90)
        u = self.u
        r, g = 3.2 * u, 11 * u
        out.append(f"{w} coords {self.iss_dot} {x - r:.0f} {y - r:.0f} {x + r:.0f} {y + r:.0f}")
        out.append(f"{w} coords {self.iss_glow} {x - g:.0f} {y - g:.0f} {x + g:.0f} {y + g:.0f}")
        out.append(f"{w} coords {self.iss_label} {x + 14 * u:.0f} {y - 10 * u:.0f}")
        behind = self.sky_rgb(y)
        self._set(out, self.iss_dot, "fill", _hex(_mix(behind, core, bright)))
        self._set(out, self.iss_glow, "fill", _hex(_mix(behind, (255, 70, 110), bright * 0.25)))
        self._set(out, self.iss_label, "fill", _hex(_mix(behind, (255, 120, 140), bright * 0.6)))
        if self.iss_trail:               # always refresh, or the previous pass's trail lingers
            pts = " ".join(f"{px:.0f} {py:.0f}" for px, py in self.iss_trail + [(x, y)])
            out.append(f"{w} coords {self.iss_trail_item} {pts}")
            self._set(out, self.iss_trail_item, "fill", _hex(_mix(behind, (255, 70, 110), bright * 0.35)))
        if not self.iss_shown:
            out.append(f"{w} itemconfigure iss -state normal")
            self.iss_shown = True


def lerp(a, b, t):
    return a + (b - a) * t
