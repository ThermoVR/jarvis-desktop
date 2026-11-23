#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.11.*"
# dependencies = ["numpy", "sgp4", "moderngl", "glfw", "python-xlib", "pillow"]
# ///
"""GPU (OpenGL) renderer for the Hey Claude face wallpaper.

face_wallpaper.py draws everything (face, weather, sky, HUD) on a Tk canvas. This file swaps
that canvas for a stand-in with the same API that keeps the items in memory and renders them
on the GPU every frame: filled shapes as triangles, every line as an antialiased capsule
(instanced quads with a distance-field shader), text from Pillow textures, painter's order kept
with a depth buffer, 4x MSAA. So all the drawing code stays in one place (face_wallpaper.py /
sky.py) and both renderers stay in sync; the Tk one remains the fallback.

Run:  ./face_wallpaper_gl.py      Env: FACE_MSAA (default 4), FACE_BLOOM (0..1, default 0.35)
      plus all the FACE_* variables of face_wallpaper.py and sky.py.
"""
import math
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MSAA = int(os.environ.get("FACE_MSAA", "4"))
BLOOM = float(os.environ.get("FACE_BLOOM", "0.35"))
LOG_PERF = Path(__file__).resolve().parent / "logs" / "perf.log"
FONT_SCALE = 0.75            # Tk font sizes in the drawing code -> pixels


def parse_color(c):
    if not c:
        return None
    c = c.strip()
    if c.startswith("#") and len(c) == 7:
        return (int(c[1:3], 16) / 255, int(c[3:5], 16) / 255, int(c[5:7], 16) / 255)
    if c.startswith("#") and len(c) == 4:
        return tuple(int(ch * 2, 16) / 255 for ch in c[1:4])
    named = {"black": (0, 0, 0), "white": (1, 1, 1), "red": (1, 0, 0)}
    return named.get(c.lower(), (0, 0, 0))


# ---- geometry helpers -------------------------------------------------------------------
def _is_convex(p):
    d = np.roll(p, -1, 0) - p
    cr = d[:, 0] * np.roll(d, -1, 0)[:, 1] - d[:, 1] * np.roll(d, -1, 0)[:, 0]
    cr = cr[np.abs(cr) > 1e-6]
    return len(cr) == 0 or np.all(cr > 0) or np.all(cr < 0)


def _earclip(p):
    """Triangulate a simple polygon (list of (x, y)); returns an (n, 2) array of triangle vertices."""
    pts = [tuple(v) for v in p]
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts)))
    if area < 0:
        pts.reverse()
    idx = list(range(len(pts)))
    out = []
    guard = 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        n = len(idx)
        for k in range(n):
            a, b, c = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % n]]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            ok = True
            for j in idx:
                q = pts[j]
                if q in (a, b, c):
                    continue
                d1 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
                d2 = (c[0] - b[0]) * (q[1] - b[1]) - (c[1] - b[1]) * (q[0] - b[0])
                d3 = (a[0] - c[0]) * (q[1] - c[1]) - (a[1] - c[1]) * (q[0] - c[0])
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    ok = False
                    break
            if ok:
                out += [a, b, c]
                del idx[k]
                break
        else:
            break                     # degenerate: give up on the rest
    if len(idx) == 3:
        out += [pts[i] for i in idx]
    return np.array(out, np.float32).reshape(-1, 2)


def _fill_tris(p):
    if len(p) >= 3:
        # drop repeated points (e.g. the crescent moon's tips appear twice), they break ear clipping
        keep = np.any(np.abs(p - np.roll(p, 1, 0)) > 0.5, 1)
        p = p[keep] if keep.any() else p[:1]
    if len(p) < 3:
        return np.zeros((0, 2), np.float32)
    if _is_convex(p):
        n = len(p)
        i = np.arange(1, n - 1)
        return np.stack([np.repeat(p[:1], n - 2, 0), p[i], p[i + 1]], 1).reshape(-1, 2).astype(np.float32)
    return _earclip(p)


def _ellipse(x0, y0, x1, y1):
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    n = int(max(12, min(72, (rx + ry) * 0.5)))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


# ---- the canvas stand-in ------------------------------------------------------------------
DEFAULTS = {
    "line": {"fill": "#000000", "width": 1.0, "capstyle": "butt"},
    "polygon": {"fill": "#000000", "outline": "", "width": 1.0},
    "oval": {"fill": "", "outline": "#000000", "width": 1.0},
    "rectangle": {"fill": "", "outline": "#000000", "width": 1.0},
    "arc": {"fill": "", "outline": "#000000", "width": 1.0},
    "text": {"fill": "#000000", "text": "", "anchor": "center", "font": ("Monospace", 12)},
}
GEOM_OPTS = {"width", "capstyle", "start", "extent", "style"}


class Item:
    __slots__ = ("id", "kind", "coords", "opts", "tags", "key", "fills", "segs", "dirty", "fill_rgb",
                 "outline_rgb")

    def __init__(self, iid, kind, coords, opts, key):
        self.id, self.kind, self.key = iid, kind, key
        self.coords = np.asarray(coords, np.float64).ravel()
        self.opts = dict(DEFAULTS[kind])
        self.tags = set()
        self.fills = self.segs = None
        self.dirty = True
        self.fill_rgb = self.outline_rgb = None
        self.set_opts(opts)

    def set_opts(self, kw):
        for k, v in kw.items():
            if k == "tags":
                self.tags = set((v,) if isinstance(v, str) else v)
                continue
            self.opts[k] = v
            if k in GEOM_OPTS:
                self.dirty = True
        self.fill_rgb = parse_color(self.opts.get("fill"))
        self.outline_rgb = parse_color(self.opts.get("outline"))

    @property
    def visible(self):
        return self.opts.get("state", "normal") != "hidden"

    def build(self):
        """Cache triangles (fill) and capsule segments (stroke) for the current coords."""
        c = self.coords
        k = self.kind
        w = max(1.0, float(self.opts.get("width", 1.0)))
        fills, segs, closed = None, None, False
        if k == "line":
            pts = c.reshape(-1, 2)
            if self.opts.get("smooth") in (True, 1, "1", "true"):
                pts = _smooth(pts)
        elif k == "polygon":
            pts = c.reshape(-1, 2)
            fills = _fill_tris(pts)
            closed = True
        elif k in ("oval", "arc"):
            x0, y0, x1, y1 = c[:4]
            if k == "arc":
                st, ex = math.radians(float(self.opts.get("start", 0))), math.radians(float(self.opts.get("extent", 90)))
                a = np.linspace(st, st + ex, 24)
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
                pts = np.stack([cx + rx * np.cos(a), cy - ry * np.sin(a)], 1)
            else:
                pts = _ellipse(x0, y0, x1, y1)
                fills = _fill_tris(pts)
                closed = True
        elif k == "rectangle":
            x0, y0, x1, y1 = c[:4]
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
            fills = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]], np.float32)
            closed = True
        else:
            pts = np.zeros((0, 2))
        if len(pts) >= 2:
            p0 = pts
            p1 = np.roll(pts, -1, 0) if closed else pts[1:]
            if not closed:
                p0 = pts[:-1]
            seg = np.concatenate([p0, p1], 1)
            seg = seg[np.any(np.abs(seg[:, :2] - seg[:, 2:]) > 1e-3, 1)]
            cap = 1.0 if (self.opts.get("capstyle") == "round" or k != "line") else 0.0
            segs = np.concatenate([seg, np.full((len(seg), 1), w / 2), np.full((len(seg), 1), cap)],
                                  1).astype(np.float32)
        self.fills = fills if fills is not None and len(fills) else None
        self.segs = segs if segs is not None and len(segs) else None
        self.dirty = False


def batch_build(dirty):
    """Rebuild many items at once: polygons and plain lines are grouped by point count and
    triangulated / split into segments with whole-array numpy ops (the head's ~400 facets
    change every frame). Anything unusual falls back to Item.build()."""
    groups = {}
    for it in dirty:
        n = len(it.coords) // 2
        if it.kind == "polygon" and n >= 3 and len(it.coords) == 2 * n:
            groups.setdefault(("polygon", n), []).append(it)
        elif it.kind == "line" and n >= 2 and len(it.coords) == 2 * n and not it.opts.get("smooth") in (True, 1, "1", "true"):
            groups.setdefault(("line", n), []).append(it)
        else:
            it.build()
    for (kind, n), items in groups.items():
        P = np.array([it.coords for it in items], np.float64).reshape(len(items), n, 2)
        hw = np.array([max(1.0, float(it.opts.get("width", 1.0))) / 2 for it in items], np.float32)
        if kind == "polygon":
            d = np.roll(P, -1, 1) - P
            d2 = np.roll(d, -1, 1)
            cr = d[..., 0] * d2[..., 1] - d[..., 1] * d2[..., 0]
            convex = ~((cr > 1e-6).any(1) & (cr < -1e-6).any(1))
            i = np.arange(1, n - 1)
            fans = np.stack([np.repeat(P[:, :1], n - 2, 1), P[:, i], P[:, i + 1]], 2) \
                .reshape(len(items), (n - 2) * 3, 2).astype(np.float32)
            seg = np.concatenate([P, np.roll(P, -1, 1)], 2)
            cap = np.ones((len(items), n), np.float32)
        else:
            convex = None
            seg = np.concatenate([P[:, :-1], P[:, 1:]], 2)
            cap = np.array([1.0 if it.opts.get("capstyle") == "round" else 0.0 for it in items],
                           np.float32)[:, None].repeat(n - 1, 1)
        live = np.any(np.abs(seg[..., :2] - seg[..., 2:]) > 1e-3, 2)
        S = np.concatenate([seg.astype(np.float32), np.broadcast_to(hw[:, None, None], seg.shape[:2] + (1,)),
                            cap[..., None]], 2)
        for j, it in enumerate(items):
            if convex is not None:
                if convex[j]:
                    it.fills = fans[j]
                else:
                    it.build()
                    continue
            else:
                it.fills = None
            sj = S[j][live[j]]
            it.segs = sj if len(sj) else None
            it.dirty = False


def _smooth(p, steps=8):
    """Tk-style smoothing: quadratic splines through the segment midpoints."""
    if len(p) < 3:
        return p
    out = [p[0]]
    t = np.linspace(0, 1, steps)[1:, None]
    for i in range(1, len(p) - 1):
        a = p[0] if i == 1 else (p[i - 1] + p[i]) / 2
        c = p[-1] if i == len(p) - 2 else (p[i] + p[i + 1]) / 2
        b = p[i]
        out.extend((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c)
    return np.array(out)


class TclShim:
    """The drawing code batches updates as Tcl scripts (`.c coords 12 ...`); run them directly."""

    def __init__(self, canvas):
        self.c = canvas

    def eval(self, script):
        c = self.c
        for line in script.split("\n"):
            tok = line.split()
            if len(tok) < 3:
                continue
            cmd = tok[1]
            if cmd == "coords":
                it = c.items.get(int(tok[2]))
                if it is not None:
                    it.coords = np.array(tok[3:], np.float64)
                    it.dirty = True
                    c._touch(it)
            elif cmd == "itemconfigure":
                kw = {}
                i = 3
                while i + 1 < len(tok):
                    kw[tok[i][1:]] = tok[i + 1]
                    i += 2
                c.itemconfigure(tok[2], **kw)
            elif cmd == "addtag":          # addtag TAG withtag ID
                for it in c._resolve(tok[4]):
                    it.tags.add(tok[2])
                    c._tag_index.setdefault(tok[2], set()).add(it.id)
            elif cmd == "dtag":            # dtag ID TAG
                for it in c._resolve(tok[2]):
                    it.tags.discard(tok[3])
                    c._tag_index.get(tok[3], set()).discard(it.id)
        return ""

    def call(self, *a):
        return ""


class GLCanvas:
    _w = ".c"

    def __init__(self, master=None, width=0, height=0, bg="#000000", **kw):
        self.width, self.height = width, height
        self.bg = parse_color(bg)
        self.items = {}
        self._tag_index = {}
        self.visible = set()            # ids of items that are not hidden
        self.top_dirty = True           # scanlines (non-text HUD layer) changed since last pack
        self._next = 1
        self._key_hi = 0.0
        self._key_lo = 0.0
        self.tk = TclShim(self)

    # -- tk API used by the drawing code
    def pack(self, **kw):
        pass

    def configure(self, **kw):
        if "bg" in kw:
            self.bg = parse_color(kw["bg"])

    config = configure

    def _new(self, kind, coords, kw):
        iid = self._next
        self._next += 1
        self._key_hi += 1
        it = Item(iid, kind, coords, kw, self._key_hi)
        self.items[iid] = it
        self._touch(it)
        for t in it.tags:
            self._tag_index.setdefault(t, set()).add(iid)
        return iid

    def create_line(self, *coords, **kw):
        return self._new("line", coords, kw)

    def create_polygon(self, *coords, **kw):
        return self._new("polygon", coords, kw)

    def create_oval(self, *coords, **kw):
        return self._new("oval", coords, kw)

    def create_rectangle(self, *coords, **kw):
        return self._new("rectangle", coords, kw)

    def create_arc(self, *coords, **kw):
        return self._new("arc", coords, kw)

    def create_text(self, x, y, **kw):
        return self._new("text", (x, y), kw)

    TOP = frozenset(("hud", "scan"))

    def _touch(self, it):
        if it.opts.get("state", "normal") == "hidden":
            self.visible.discard(it.id)
        else:
            self.visible.add(it.id)
        if it.kind != "text" and not self.TOP.isdisjoint(it.tags):
            self.top_dirty = True

    def _resolve(self, target):
        if isinstance(target, int):
            it = self.items.get(target)
            return [it] if it else []
        s = str(target)
        if s.isdigit():
            it = self.items.get(int(s))
            return [it] if it else []
        if s == "all":
            return list(self.items.values())
        return [self.items[i] for i in self._tag_index.get(s, ()) if i in self.items]

    def coords(self, target, *pts):
        for it in self._resolve(target):
            if len(pts) == 1 and not np.isscalar(pts[0]):
                pts = tuple(pts[0])
            it.coords = np.asarray(pts, np.float64).ravel()
            it.dirty = True
            self._touch(it)

    def itemconfigure(self, target, **kw):
        for it in self._resolve(target):
            if "tags" in kw:
                for t in it.tags:
                    self._tag_index.get(t, set()).discard(it.id)
            it.set_opts(kw)
            if "tags" in kw:
                for t in it.tags:
                    self._tag_index.setdefault(t, set()).add(it.id)
            self._touch(it)

    itemconfig = itemconfigure

    def delete(self, target):
        for it in self._resolve(target):
            for t in it.tags:
                self._tag_index.get(t, set()).discard(it.id)
            del self.items[it.id]
            self.visible.discard(it.id)
            if it.kind != "text" and not self.TOP.isdisjoint(it.tags):
                self.top_dirty = True

    def tag_raise(self, target, above=None):
        if target in self.TOP:
            return                      # the renderer always draws the HUD layer on top
        for it in sorted(self._resolve(target), key=lambda i: i.key):
            self._key_hi += 1
            it.key = self._key_hi

    def tag_lower(self, target, below=None):
        for it in sorted(self._resolve(target), key=lambda i: i.key, reverse=True):
            self._key_lo -= 1
            it.key = self._key_lo

    def set_facets(self, ids, vis, pts, cols, changed):
        """Fast path for the head's facets: coords for visible ones, colours only where changed
        (cols rows are fill rgb + outline rgb as 0..255 ints, -1 = hidden)."""
        items = self.items
        for i in np.flatnonzero(vis):
            it = items.get(ids[i])
            if it is not None:
                it.coords = pts[i]
                it.dirty = True
        for i in np.flatnonzero(changed):
            it = items.get(ids[i])
            if it is None:
                continue
            c = cols[i]
            if c[0] < 0:
                it.opts["state"] = "hidden"
                self.visible.discard(it.id)
            else:
                it.opts["state"] = "normal"
                self.visible.add(it.id)
                it.fill_rgb = (c[0] / 255, c[1] / 255, c[2] / 255)
                it.outline_rgb = (c[3] / 255, c[4] / 255, c[5] / 255)

    def find_all(self):
        return list(self.items)

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height


class FakeRoot:
    """Stand-in for tk.Tk: geometry and the after() timer; the GL loop drives it."""

    def __init__(self, className=None, **kw):
        from Xlib import display
        d = display.Display()
        scr = d.screen()
        self.sw, self.sh = scr.width_in_pixels, scr.height_in_pixels
        d.close()
        self.timers = []
        self.geom = (self.sw, self.sh, 0, 0)

    def title(self, *a):
        pass

    def geometry(self, spec):
        m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", spec)
        if m:
            self.geom = tuple(int(v) for v in m.groups())

    def configure(self, **kw):
        pass

    def attributes(self, *a):
        pass

    def overrideredirect(self, *a):
        pass

    def lower(self, *a):
        pass

    def winfo_screenwidth(self):
        return self.sw

    def winfo_screenheight(self):
        return self.sh

    def after(self, ms, fn, *args):
        self.timers.append((time.monotonic() + ms / 1000, fn, args))

    def mainloop(self):
        raise RuntimeError("driven by the GL loop")


fake_tk = types.SimpleNamespace(Tk=FakeRoot, Canvas=GLCanvas, TclError=Exception)

# ---- shaders --------------------------------------------------------------------------------
FILL_VS = """
#version 330
uniform vec2 screen;
in vec3 pos; in vec3 col;
out vec3 v_col;
void main() {
    gl_Position = vec4(pos.x / screen.x * 2.0 - 1.0, 1.0 - pos.y / screen.y * 2.0, pos.z, 1.0);
    v_col = col;
}"""
FILL_FS = """
#version 330
in vec3 v_col; out vec4 frag;
void main() { frag = vec4(v_col, 1.0); }"""

# one instance per line segment: a quad around it, the fragment shader cuts the capsule
SEG_VS = """
#version 330
uniform vec2 screen;
in vec4 seg; in vec2 hw_cap; in float z; in vec3 col;
out vec2 v_p; flat out vec4 v_seg; flat out vec2 v_hc; flat out vec3 v_col;
void main() {
    vec2 a = seg.xy, b = seg.zw;
    vec2 d = b - a; float L = length(d);
    vec2 t = L > 1e-4 ? d / L : vec2(1.0, 0.0);
    vec2 n = vec2(-t.y, t.x);
    float r = hw_cap.x + 1.0;
    int id = gl_VertexID;
    vec2 along = (id == 0 || id == 2) ? a - t * r : b + t * r;
    vec2 p = along + n * ((id < 2) ? -r : r);
    v_p = p; v_seg = seg; v_hc = hw_cap; v_col = col;
    gl_Position = vec4(p.x / screen.x * 2.0 - 1.0, 1.0 - p.y / screen.y * 2.0, z, 1.0);
}"""
SEG_FS = """
#version 330
in vec2 v_p; flat in vec4 v_seg; flat in vec2 v_hc; flat in vec3 v_col;
out vec4 frag;
void main() {
    vec2 a = v_seg.xy, b = v_seg.zw, d = b - a;
    float L2 = max(dot(d, d), 1e-6);
    float h = dot(v_p - a, d) / L2;
    float dist;
    if (v_hc.y > 0.5) {
        dist = length(v_p - a - d * clamp(h, 0.0, 1.0));           // round caps: capsule
    } else {
        float L = sqrt(L2);
        float along = max(-h * L, (h - 1.0) * L);                    // butt caps: flat ends
        dist = max(length(v_p - a - d * h), along + v_hc.x);
        dist = max(dist, abs(dot(v_p - a, vec2(-d.y, d.x)) / L));
    }
    float hw = max(v_hc.x, 0.5);
    float cov = clamp(hw - dist + 0.5, 0.0, 1.0);
    if (cov <= 0.0) discard;
    frag = vec4(v_col, cov);
}"""
TEXT_VS = """
#version 330
uniform vec2 screen;
in vec2 pos; in vec2 uv;
out vec2 v_uv;
uniform float z;
void main() {
    gl_Position = vec4(pos.x / screen.x * 2.0 - 1.0, 1.0 - pos.y / screen.y * 2.0, z, 1.0);
    v_uv = uv;
}"""
TEXT_FS = """
#version 330
uniform sampler2D tex; uniform vec3 col;
in vec2 v_uv; out vec4 frag;
void main() { float a = texture(tex, v_uv).r; if (a < 0.01) discard; frag = vec4(col, a); }"""
QUAD_VS = """
#version 330
in vec2 pos; out vec2 v_uv;
void main() { v_uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }"""
BLUR_FS = """
#version 330
uniform sampler2D tex; uniform vec2 dir; uniform float thresh;
in vec2 v_uv; out vec4 frag;
const float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
vec3 src(vec2 uv) { vec3 c = texture(tex, uv).rgb; return max(c - thresh, 0.0); }
void main() {
    vec3 s = src(v_uv) * w[0];
    for (int i = 1; i < 5; i++) { s += (src(v_uv + dir * float(i)) + src(v_uv - dir * float(i))) * w[i]; }
    frag = vec4(s, 1.0);
}"""
COMPOSE_FS = """
#version 330
uniform sampler2D scene; uniform sampler2D glow; uniform float amount;
in vec2 v_uv; out vec4 frag;
void main() { frag = vec4(texture(scene, v_uv).rgb + texture(glow, v_uv).rgb * amount, 1.0); }"""


class Renderer:
    def __init__(self, canvas, W, H, X=0, Y=0, head=None):
        import glfw
        import moderngl
        self.glfw, self.mgl = glfw, moderngl
        self.c, self.W, self.H = canvas, W, H
        if not glfw.init():
            raise SystemExit("glfw init failed")
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        try:
            glfw.window_hint_string(glfw.X11_CLASS_NAME, "hey-claude-face")
            glfw.window_hint_string(glfw.X11_INSTANCE_NAME, "hey-claude-face")
        except Exception:
            pass
        self.win = glfw.create_window(W, H, "Hey Claude Face", None, None)
        self._make_desktop_window(glfw.get_x11_window(self.win))
        glfw.set_window_pos(self.win, X, Y)
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)
        glfw.show_window(self.win)
        ctx = self.ctx = moderngl.create_context()
        print("GL:", ctx.info["GL_RENDERER"], flush=True)
        self.fill_prog = ctx.program(vertex_shader=FILL_VS, fragment_shader=FILL_FS)
        self.seg_prog = ctx.program(vertex_shader=SEG_VS, fragment_shader=SEG_FS)
        self.text_prog = ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        for p in (self.fill_prog, self.seg_prog, self.text_prog):
            p["screen"].value = (W, H)
        samples = min(MSAA, ctx.max_samples)
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=samples)],
                                    depth_attachment=ctx.depth_renderbuffer((W, H), samples=samples))
        self.scene_tex = ctx.texture((W, H), 4)
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex])
        hw, hh = W // 4, H // 4                                  # bloom at quarter resolution
        self.blur_tex = [ctx.texture((hw, hh), 4) for _ in range(2)]
        for t in [self.scene_tex] + self.blur_tex:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.blur_fbo = [ctx.framebuffer(color_attachments=[t]) for t in self.blur_tex]
        self.blur_prog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=BLUR_FS)
        self.compose_prog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=COMPOSE_FS)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.blur_vao = ctx.vertex_array(self.blur_prog, [(quad, "2f", "pos")])
        self.compose_vao = ctx.vertex_array(self.compose_prog, [(quad, "2f", "pos")])
        self.fill_buf = ctx.buffer(reserve=1 << 20, dynamic=True)
        self.seg_buf = ctx.buffer(reserve=1 << 20, dynamic=True)
        # orphan() keeps the same GL buffer, so these vertex arrays stay valid when buffers grow
        self.fill_vao = ctx.vertex_array(self.fill_prog, [(self.fill_buf, "3f 3f", "pos", "col")])
        self.seg_vao = ctx.vertex_array(self.seg_prog, [(self.seg_buf, "4f 2f 1f 3f /i", "seg", "hw_cap", "z", "col")])
        self.text_buf = ctx.buffer(reserve=4 * 4 * 4)
        self.text_vao = ctx.vertex_array(self.text_prog, [(self.text_buf, "2f 2f", "pos", "uv")])
        self.head_r = None
        self.top_cache = None
        if head is not None:
            from head_gl import HeadRenderer
            self.head_r = HeadRenderer(ctx, W, H, head, SEG_FS)
        self.textures = {}               # (text, font) -> (texture, w, h, last_used)
        self.fonts = {}
        self.stats = {"t0": time.monotonic(), "n": 0, "ms": []}

    def _make_desktop_window(self, xid):
        """Desktop-type window: no decorations, below all normal windows, not in alt-tab."""
        from Xlib import display, Xatom
        d = display.Display()
        w = d.create_resource_object("window", xid)
        w.change_property(d.intern_atom("_NET_WM_WINDOW_TYPE"), Xatom.ATOM, 32,
                          [d.intern_atom("_NET_WM_WINDOW_TYPE_DESKTOP")])
        w.set_wm_class("hey-claude-face", "hey-claude-face")
        d.flush()
        d.close()

    def _ensure(self, buf, nbytes):
        if buf.size < nbytes:
            buf.orphan(max(nbytes, buf.size * 2))
        return buf

    # -- text through Pillow
    def _font(self, spec):
        from PIL import ImageFont
        fam, size = spec[0], int(spec[1]) if len(spec) > 1 else 12
        bold = len(spec) > 2 and "bold" in str(spec[2])
        px = max(8, int(abs(size) * FONT_SCALE))
        key = (px, bold)
        if key not in self.fonts:
            import subprocess
            path = subprocess.run(["fc-match", "-f", "%{file}", f"Monospace:{'bold' if bold else 'regular'}"],
                                  capture_output=True, text=True).stdout.strip()
            try:
                self.fonts[key] = ImageFont.truetype(path, px)
            except OSError:
                self.fonts[key] = ImageFont.load_default()
        return self.fonts[key]

    def _text_tex(self, text, font):
        key = (text, tuple(font) if isinstance(font, (tuple, list)) else font)
        hit = self.textures.get(key)
        now = time.monotonic()
        if hit:
            self.textures[key] = hit[:3] + (now,)
            return hit
        from PIL import Image, ImageDraw
        f = self._font(font if isinstance(font, (tuple, list)) else ("Monospace", 12))
        l, t, r, b = f.getbbox(text or " ")
        asc, desc = f.getmetrics()
        w, h = max(1, r - min(l, 0) + 2), asc + desc + 2
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((1 - min(l, 0), 1), text, font=f, fill=255)
        tex = self.ctx.texture((w, h), 1, img.tobytes())
        tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        entry = (tex, w, h, now)
        self.textures[key] = entry
        if len(self.textures) > 64:                      # drop stale clock strings etc.
            for k2, e in sorted(self.textures.items(), key=lambda kv: kv[1][3])[:16]:
                e[0].release()
                del self.textures[k2]
        return entry

    # -- one frame
    def _pack(self, vis, zlo, zhi):
        """Canvas items in painter's order -> (fill vertex array, segment instance array, texts),
        depth mapped into the slice (zlo, zhi)."""
        n = len(vis) + 2
        span = zhi - zlo
        fills, fcount, fz, fcol, segs, scount, sz, scol, texts = [], [], [], [], [], [], [], [], []
        for rank, it in enumerate(vis):
            if it.kind == "text":
                if it.fill_rgb and it.opts.get("text"):
                    texts.append((it, zhi - (rank + 1.5) / n * span))
                continue
            if it.dirty:
                it.build()
            z = zhi - (rank + 1) / n * span
            if it.fills is not None and it.fill_rgb is not None:
                fills.append(it.fills)
                fcount.append(len(it.fills))
                fz.append(z)
                fcol.append(it.fill_rgb)
            stroke = it.fill_rgb if it.kind == "line" else it.outline_rgb
            if it.segs is not None and stroke is not None:
                segs.append(it.segs)
                scount.append(len(it.segs))
                sz.append(z - 0.4 / n * span)
                scol.append(stroke)
        F = S = None
        if fills:
            P = np.concatenate(fills)
            F = np.empty((len(P), 6), np.float32)
            F[:, :2] = P
            F[:, 2] = np.repeat(np.array(fz, np.float32), fcount)
            F[:, 3:] = np.repeat(np.array(fcol, np.float32), fcount, 0)
        if segs:
            Q = np.concatenate(segs)
            S = np.empty((len(Q), 10), np.float32)
            S[:, :6] = Q
            S[:, 6] = np.repeat(np.array(sz, np.float32), scount)
            S[:, 7:] = np.repeat(np.array(scol, np.float32), scount, 0)
        return F, S, texts

    def _submit(self, F, S, texts):
        ctx = self.ctx
        ctx.enable(self.mgl.DEPTH_TEST)
        ctx.depth_func = "<"
        if F is not None:
            self._ensure(self.fill_buf, F.nbytes).write(F.tobytes())
            self.fill_vao.render(self.mgl.TRIANGLES, vertices=len(F))
        if S is not None:
            self._ensure(self.seg_buf, S.nbytes).write(S.tobytes())
            ctx.enable_direct(0x809E)                        # GL_SAMPLE_ALPHA_TO_COVERAGE
            self.seg_vao.render(self.mgl.TRIANGLE_STRIP, vertices=4, instances=len(S))
            ctx.disable_direct(0x809E)
        if texts:
            ctx.enable(self.mgl.BLEND)
            ctx.blend_func = self.mgl.SRC_ALPHA, self.mgl.ONE_MINUS_SRC_ALPHA
            ctx.depth_func = "<="
            for it, z in texts:
                tex, w, h, _ = self._text_tex(str(it.opts["text"]), it.opts.get("font"))
                x, y = it.coords[0], it.coords[1]
                anchor = it.opts.get("anchor", "center")
                x0 = x - w / 2 if anchor in ("center", "n", "s") else (x if "w" in anchor else x - w)
                y0 = y - h / 2 if anchor in ("center", "w", "e") else (y if "n" in anchor else y - h)
                q = np.array([x0, y0, 0, 0, x0 + w, y0, 1, 0, x0, y0 + h, 0, 1, x0 + w, y0 + h, 1, 1], np.float32)
                self.text_buf.write(q.tobytes())
                tex.use(0)
                self.text_prog["tex"].value = 0
                self.text_prog["col"].value = it.fill_rgb
                self.text_prog["z"].value = z
                self.text_vao.render(self.mgl.TRIANGLE_STRIP)
            ctx.disable(self.mgl.BLEND)

    def _clear_depth(self):
        self.msaa.color_mask = (False, False, False, False)
        self.msaa.clear(depth=1.0)
        self.msaa.color_mask = (True, True, True, True)

    def render(self):
        t0 = time.monotonic()
        ctx, c = self.ctx, self.c
        items = c.items
        vis = [items[i] for i in c.visible]
        top_tags = c.TOP
        low, hud = [], []
        for it in vis:
            if top_tags.isdisjoint(it.tags):
                low.append(it)
            elif it.kind == "text":
                hud.append(it)
        low.sort(key=lambda i: i.key)
        batch_build([it for it in low if it.dirty and it.kind != "text"])
        self.msaa.use()
        bg = c.bg or (0, 0, 0)
        ctx.clear(*bg, 1.0, depth=1.0)
        # depth slices (NDC, smaller = in front): scene 0.25..1, head facets -0.2..0.2,
        # head decorations -0.21..-0.5, HUD + scanlines -0.55..-1
        self._submit(*self._pack(low, 0.25, 1.0))
        if self.head_r is not None:
            self.head_r.draw()
        # HUD + scanlines on top. Scanlines (static) are packed only when they change; Tk keeps
        # them above the HUD text, so the text gets a depth just behind them.
        if c.top_dirty or self.top_cache is None:
            top = sorted((it for it in vis if it.kind != "text" and not top_tags.isdisjoint(it.tags)),
                         key=lambda it: ("scan" in it.tags, it.key))
            batch_build([it for it in top if it.dirty])
            self.top_cache = self._pack(top, -1.0, -0.6)[:2]
            c.top_dirty = False
        self._submit(*self.top_cache, [(it, -0.58) for it in sorted(hud, key=lambda i: i.key)])
        ctx.disable(self.mgl.DEPTH_TEST)
        if BLOOM > 0:
            # resolve, blur the bright parts at quarter resolution, add them back
            ctx.copy_framebuffer(self.scene_fbo, self.msaa)
            hw, hh = self.blur_tex[0].size
            self.blur_prog["thresh"].value = 0.18
            self.blur_fbo[0].use()
            self.scene_tex.use(0)
            self.blur_prog["tex"].value = 0
            self.blur_prog["dir"].value = (1.6 / hw, 0)
            self.blur_vao.render(self.mgl.TRIANGLE_STRIP)
            self.blur_prog["thresh"].value = 0.0
            for i in range(2):
                self.blur_fbo[1].use()
                self.blur_tex[0].use(0)
                self.blur_prog["dir"].value = (0, 1.6 / hh)
                self.blur_vao.render(self.mgl.TRIANGLE_STRIP)
                self.blur_fbo[0].use()
                self.blur_tex[1].use(0)
                self.blur_prog["dir"].value = (1.6 / hw * (i + 1), 0)
                self.blur_vao.render(self.mgl.TRIANGLE_STRIP)
            ctx.screen.use()
            self.scene_tex.use(0)
            self.blur_tex[0].use(1)
            self.compose_prog["scene"].value = 0
            self.compose_prog["glow"].value = 1
            self.compose_prog["amount"].value = BLOOM
            self.compose_vao.render(self.mgl.TRIANGLE_STRIP)
        else:
            ctx.copy_framebuffer(ctx.screen, self.msaa)
        self.glfw.swap_buffers(self.win)
        self._log((time.monotonic() - t0) * 1000, len(vis))

    def _log(self, ms, n_items):
        s = self.stats
        s["n"] += 1
        s["ms"].append(ms)
        span = time.monotonic() - s["t0"]
        if span < 5 or os.environ.get("FACE_PERF", "1") == "0":
            return
        d = sorted(s["ms"])
        line = (f"{time.strftime('%H:%M:%S')} GL render fps={s['n'] / span:5.1f} cpu_ms avg={sum(d) / len(d):5.1f} "
                f"p95={d[int(len(d) * .95)]:5.1f} max={d[-1]:5.1f} visible_items={n_items}")
        try:
            with open(LOG_PERF, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        self.stats = {"t0": time.monotonic(), "n": 0, "ms": []}


def main():
    import face_wallpaper as fw
    from head_gl import HeadGPU
    fw.tk = fake_tk                         # the face now draws into GLCanvas
    gpu_head = os.environ.get("FACE_GPU_HEAD", "1") != "0"
    if gpu_head:
        orig_frame = fw.Face.frame

        def frame(self, t):
            self.gpu.frame = None           # no head this frame unless _draw_head runs
            orig_frame(self, t)

        fw.Face.frame = frame
        fw.Face._draw_head = lambda self, *a: self.gpu.draw(*a)
    fw.pin_cpus()
    face = fw.Face()
    head = None
    if gpu_head:
        head = face.gpu = HeadGPU(face)
    root = face.root
    W, H, X, Y = root.geom
    r = Renderer(face.c, W, H, X, Y, head=head)
    glfw = r.glfw
    try:
        while not glfw.window_should_close(r.win):
            now = time.monotonic()
            due = [tm for tm in root.timers if tm[0] <= now]
            if due:
                root.timers = [tm for tm in root.timers if tm[0] > now]
                for _, fn, args in due:
                    fn(*args)
                r.render()
            nxt = min((tm[0] for tm in root.timers), default=now + 0.05)
            glfw.wait_events_timeout(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        pass
    glfw.terminate()


if __name__ == "__main__":
    main()
