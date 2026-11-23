"""The Jarvis head on the GPU.

face_wallpaper.py computes the head with numpy every frame (surface, rotation, projection,
per-facet shading) and hands thousands of shapes to the canvas. Here the same maths runs in
shaders: the mask and hair are static meshes in surface coordinates (y, phi), and a vertex /
geometry shader does surface -> camera -> screen plus the flat facet shading. The decorations
(eyes, mouth, chip, temple discs, lips, undercut) are sent as short lists of surface points;
the GPU projects them and draws them as glowing capsules. Python only computes a few dozen
numbers per frame.

KEEP IN SYNC with face_wallpaper.py (_surface, _draw_head and helpers): the Tk version is the
reference and the fallback.
"""
import math
import random

import numpy as np

BG = np.array((3, 1, 6), np.float64)
HOT = (255, 255, 255)

# ---- shared GLSL: head surface, camera, projection -------------------------------------------
GLSL_HEAD = """
uniform vec2 screen;
uniform float PY[12]; uniform float PW[12]; uniform float PF[12]; uniform float PB[12];
uniform float fem;
uniform vec4 rot;          // cos yaw, sin yaw, cos pitch, sin pitch
uniform vec2 hc;           // head centre on screen (px)
uniform float scale; uniform float camd;

float prof(float yi, float p[12]) {
    for (int i = 0; i < 11; i++) {
        if (yi >= PY[i + 1]) {
            float t = (PY[i] - yi) / (PY[i] - PY[i + 1]);
            return mix(p[i], p[i + 1], t);
        }
    }
    return p[11];
}
float g(float x, float y, float cx, float cy, float sx, float sy) {
    float a = (x - cx) / sx, b = (y - cy) / sy;
    return exp(-a * a - b * b);
}
vec3 surface(float y, float phi, float jaw) {
    float yi = clamp(y, -1.0, 1.0);
    float w = prof(yi, PW), zf = prof(yi, PF), zb = prof(yi, PB);
    float sp = sin(phi), cp = cos(phi);
    float x = w * sign(sp) * pow(abs(sp), 0.9);
    float z = (cp > 0.0 ? zf : zb) * sign(cp) * pow(abs(cp), 0.75);
    float front = clamp(cp * 2.5, 0.0, 1.0);
    float m = 1.0 - fem;
    z += front * (mix(0.16, 0.12, fem) * g(x, y, 0.0, -0.13, mix(0.07, 0.055, fem), 0.17)
        + 0.05 * m * g(x, y, 0.0, -0.26, 0.13, 0.05)
        + mix(0.09, 0.035, fem) * g(x, y, 0.0, 0.27, 0.5, 0.05) + 0.03 * g(x, y, 0.0, 0.22, 0.1, 0.05)
        - 0.11 * g(x, y, -0.27, 0.15, 0.15, 0.08) - 0.11 * g(x, y, 0.27, 0.15, 0.15, 0.08)
        + 0.07 * g(x, y, -0.4, -0.08, 0.1, 0.08) + 0.07 * g(x, y, 0.4, -0.08, 0.1, 0.08)
        - 0.04 * g(x, y, -0.3, -0.3, 0.1, 0.12) - 0.04 * g(x, y, 0.3, -0.3, 0.1, 0.12)
        + 0.04 * fem * g(x, y, -0.38, 0.0, 0.1, 0.07) + 0.04 * fem * g(x, y, 0.38, 0.0, 0.1, 0.07)
        + mix(0.05, 0.075, fem) * g(x, y, 0.0, -0.45, mix(0.17, 0.15, fem), mix(0.07, 0.085, fem))
        + mix(0.06, 0.035, fem) * g(x, y, 0.0, -0.82, 0.15, 0.1));
    if (jaw != 0.0) {
        float wj = front * clamp((-0.46 - y) / 0.05, 0.0, 1.0) * clamp(1.0 - abs(x) / 0.6, 0.0, 1.0);
        y -= jaw * 0.12 * wj;
    }
    return vec3(x, y, z);
}
vec3 cam(vec3 P) {
    float x1 = P.x * rot.x + P.z * rot.y;
    float z1 = -P.x * rot.y + P.z * rot.x;
    return vec3(x1, P.y * rot.z - z1 * rot.w, P.y * rot.w + z1 * rot.z);
}
vec2 proj(vec3 C) {
    float f = camd / (camd - C.z);
    return vec2(hc.x + scale * C.x * f, hc.y - scale * C.y * f);
}
vec4 clip_pos(vec2 p, float depth) {
    return vec4(p.x / screen.x * 2.0 - 1.0, 1.0 - p.y / screen.y * 2.0, depth, 1.0);
}
bool visible(vec3 P) {          // surface point faces the camera (outward ~ horizontal from the axis)
    vec3 C = cam(P), O = cam(P * vec3(1.0, 0.25, 1.0));
    vec3 view = vec3(0.0, 0.0, camd) - C;
    return dot(O, view) > 0.15 * length(view);
}
"""

# ---- faceted mask / hair: geometry shader shades each triangle and adds its outline ---------
FACET_VS = """
#version 330
""" + GLSL_HEAD + """
in vec2 yp; in vec2 extra;       // mask: extra.x = eye weight; hair: extra = (sway offset, puff)
uniform float jaw; uniform float t; uniform int hair;
out vec3 v_world; out vec3 v_cam; out float v_eyew;
void main() {
    vec3 P;
    if (hair == 1) {
        float y = yp.x;
        float sway = 0.012 * sin(t * 1.1 + extra.x + y * 2.5) * clamp((0.3 - y) / 1.5, 0.0, 1.0);
        P = surface(clamp(y, -1.0, 1.0), yp.y + sway, 0.0) * 1.07;
        P.y = y;
        P *= extra.y;
        v_eyew = 0.0;
    } else {
        P = surface(yp.x, yp.y, jaw);
        v_eyew = extra.x;
    }
    v_world = P;
    v_cam = cam(P);
}"""
FACET_GS = """
#version 330
""" + GLSL_HEAD + """
layout(triangles) in;
layout(triangle_strip, max_vertices = 15) out;
in vec3 v_world[]; in vec3 v_cam[]; in float v_eyew[];
uniform int hair; uniform float t; uniform int thinking;
uniform float eye_glow; uniform vec3 prim; uniform vec3 sec; uniform float kk; uniform float edge_w;
flat out vec3 g_col;
float dep(vec3 C) { return -C.z * 0.15; }
void main() {
    vec3 A = v_cam[0], B = v_cam[1], D = v_cam[2];
    vec3 n = cross(B - A, D - A);
    vec3 cen = (A + B + D) / 3.0;
    if (hair == 1) {
        vec3 Vc = (v_world[0] + v_world[1] + v_world[2]) / 3.0;
        vec3 radial = cam(vec3(Vc.x, max(Vc.y - 0.3, 0.0), Vc.z));
        n *= sign(dot(n, radial) + 1e-9);
    } else {
        n *= sign(dot(n, cen));
    }
    n /= length(n) + 1e-9;
    vec3 view = normalize(vec3(0.0, 0.0, camd) - cen);
    float facing = dot(n, view);
    if (facing < 0.0) return;
    vec3 L = normalize(vec3(-0.45, 0.65, 0.6));
    vec3 fill, edge;
    if (hair == 1) {
        float dl = clamp(abs(dot(n, L)), 0.0, 1.0);
        float rim = pow(clamp(1.0 - abs(facing), 0.0, 1.0), 2.0);
        fill = 14.0 + prim * (0.07 + 0.2 * dl * dl) + sec * 0.05;
        edge = fill + (prim - fill) * (0.25 + 0.6 * rim);
    } else {
        float diff = clamp(dot(n, L), 0.0, 1.0);
        float under = clamp(dot(n, vec3(0.0, -0.8, 0.6)), 0.0, 1.0);
        float rim = pow(clamp(1.0 - facing, 0.0, 1.0), 2.0);
        float cy = (v_world[0].y + v_world[1].y + v_world[2].y) / 3.0;
        float band;
        if (thinking == 1) {
            float b1 = (cy - (1.2 - fract(t * 0.9) * 2.6)) / 0.09;
            float b2 = (cy - (1.2 - fract(t * 0.9 + 0.5) * 2.6)) / 0.09;
            band = exp(-b1 * b1) + exp(-b2 * b2);
        } else {
            float b = (cy - (1.2 - fract(t * 0.25) * 3.0)) / 0.05;
            band = 0.15 * exp(-b * b);
        }
        float ew = v_eyew[0] * eye_glow;
        vec3 base = vec3(9.0, 8.0, 13.0) + vec3(38.0, 38.0, 46.0) * diff * diff;
        fill = base + sec * 0.10 * under + prim * 0.22 * ew + prim * 0.4 * band;
        float tt = min(1.0, 0.05 + 0.7 * pow(rim, 1.3) + band);
        edge = fill + (prim - fill) * tt;
    }
    vec3 bg = vec3(3.0, 1.0, 6.0);
    fill = clamp(bg + (fill - bg) * kk, 0.0, 255.0) / 255.0;
    edge = clamp(bg + (edge - bg) * kk, 0.0, 255.0) / 255.0;
    vec2 p[3];
    for (int i = 0; i < 3; i++) p[i] = proj(v_cam[i]);
    for (int i = 0; i < 3; i++) { g_col = fill; gl_Position = clip_pos(p[i], dep(v_cam[i])); EmitVertex(); }
    EndPrimitive();
    float hw = max(0.5, edge_w * 0.5);
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        vec2 d = p[j] - p[i];
        float len = length(d);
        if (len < 1e-3) continue;
        vec2 tdir = d / len, nrm = vec2(-tdir.y, tdir.x) * hw;
        vec2 a = p[i] - tdir * hw * 0.5, b = p[j] + tdir * hw * 0.5;
        float da = dep(v_cam[i]) - 0.004, db = dep(v_cam[j]) - 0.004;
        g_col = edge; gl_Position = clip_pos(a - nrm, da); EmitVertex();
        g_col = edge; gl_Position = clip_pos(a + nrm, da); EmitVertex();
        g_col = edge; gl_Position = clip_pos(b - nrm, db); EmitVertex();
        g_col = edge; gl_Position = clip_pos(b + nrm, db); EmitVertex();
        EndPrimitive();
    }
}"""
FACET_FS = """
#version 330
flat in vec3 g_col; out vec4 frag;
void main() { frag = vec4(g_col, 1.0); }"""

# ---- decorations: glowing capsules between two surface points, and filled shapes ------------
DECO_SEG_VS = """
#version 330
""" + GLSL_HEAD + """
in vec3 a; in vec3 b;              // (y, phi, jaw) of both ends
in vec4 shv;                       // shell, half width (px), visibility test, depth
in vec3 col;
out vec2 v_p; flat out vec4 v_seg; flat out vec2 v_hc; flat out vec3 v_col;
void main() {
    vec3 Pa = surface(a.x, a.y, a.z) * shv.x, Pb = surface(b.x, b.y, b.z) * shv.x;
    if (shv.z > 0.5 && !(visible(Pa) && visible(Pb))) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    vec2 pa = proj(cam(Pa)), pb = proj(cam(Pb));
    vec2 d = pb - pa; float L = length(d);
    vec2 t = L > 1e-4 ? d / L : vec2(1.0, 0.0);
    vec2 n = vec2(-t.y, t.x);
    float r = shv.y + 1.0;
    int id = gl_VertexID;
    vec2 along = (id == 0 || id == 2) ? pa - t * r : pb + t * r;
    vec2 p = along + n * ((id < 2) ? -r : r);
    v_p = p; v_seg = vec4(pa, pb); v_hc = vec2(shv.y, 1.0); v_col = col;
    gl_Position = clip_pos(p, shv.w);
}"""
DECO_FILL_VS = """
#version 330
""" + GLSL_HEAD + """
in vec3 ypj; in vec2 shz; in vec3 col;
out vec3 v_col;
void main() {
    vec3 P = surface(ypj.x, ypj.y, ypj.z) * shz.x;
    gl_Position = clip_pos(proj(cam(P)), shz.y);
    v_col = col;
}"""
DECO_FILL_FS = """
#version 330
in vec3 v_col; out vec4 frag;
void main() { frag = vec4(v_col, 1.0); }"""


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_rgb(a, b, t):
    return tuple(lerp(a[i], b[i], t) for i in range(3))


def col01(rgb, k=1.0):
    """Same as face_wallpaper.hexcol, as floats 0..1."""
    kk = max(0.0, min(1.0, k))
    return tuple((BG[i] + (rgb[i] - BG[i]) * kk) / 255.0 for i in range(3))


def densify(pts, per=8):
    out = []
    for (y0, p0), (y1, p1) in zip(pts, pts[1:]):
        for i in range(per):
            f = i / per
            out.append((y0 + (y1 - y0) * f, p0 + (p1 - p0) * f))
    out.append(pts[-1])
    return np.array(out)


def slide(Y, PHI, frac, half=1.6, n=6):
    """Sub-path centred at fraction frac, interpolated (same as Face._slide, in surface coords)."""
    m = len(Y)
    pos = frac * (m + 2 * half) - half
    idx = np.clip(np.linspace(pos - half, pos + half, n), 0, m - 1)
    if idx[-1] - idx[0] < 0.2:
        return None
    xs = np.arange(m)
    return np.interp(idx, xs, Y), np.interp(idx, xs, PHI)


class Deco:
    """Collects one frame of decorations in draw order (later = on top); finish() turns them
    into GPU instance arrays with a handful of whole-array numpy ops."""

    def __init__(self):
        self.specs, self.fills, self.order = [], [], 0

    def _z(self):
        self.order += 1
        return -0.21 - self.order * 1e-4

    def line(self, Y, PHI, J, shell, col, hw, vis, closed=False):
        self.specs.append((Y, PHI, J, shell, col, hw, vis, closed, self._z()))

    def dot(self, y, phi, shell, radius, col, vis):
        self.line((y, y), (phi, phi), 0.0, shell, col, radius, vis)

    def finish(self):
        sp = self.specs
        if not sp:
            return None
        Ys = [np.asarray(s[0], np.float32).ravel() for s in sp]
        counts = np.array([len(y) for y in Ys])
        Y = np.concatenate(Ys)
        P = np.concatenate([np.asarray(s[1], np.float32).ravel() for s in sp])
        J = np.concatenate([np.asarray(s[2], np.float32).ravel() if np.ndim(s[2]) else np.full(c, s[2], np.float32)
                            for s, c in zip(sp, counts)])
        ends = np.cumsum(counts)
        starts = ends - counts
        valid = np.ones(len(Y) - 1, bool)
        valid[ends[:-1] - 1] = False
        a = np.flatnonzero(valid)
        b = a + 1
        closed = np.array([s[7] for s in sp]) & (counts > 2)
        if closed.any():
            a = np.concatenate([a, ends[closed] - 1])
            b = np.concatenate([b, starts[closed]])
        line_of = np.repeat(np.arange(len(sp)), counts)[a]
        attrs = np.array([(s[3], s[5], s[6], s[8], *s[4]) for s in sp], np.float32)[line_of]
        out = np.empty((len(a), 13), np.float32)
        out[:, 0], out[:, 1], out[:, 2] = Y[a], P[a], J[a]
        out[:, 3], out[:, 4], out[:, 5] = Y[b], P[b], J[b]
        out[:, 6:13] = attrs
        return out

    def fill(self, Y, PHI, J, shell, col):
        """Filled shape: a fan around the centroid (all shapes here are star-shaped)."""
        Y, PHI = np.asarray(Y, float), np.asarray(PHI, float)
        J = np.broadcast_to(np.asarray(J, float), Y.shape)
        n = len(Y)
        cy, cp, cj = Y.mean(), PHI.mean(), J.mean()
        i = np.arange(n)
        j = (i + 1) % n
        v = np.empty((n * 3, 8), np.float32)
        v[0::3, 0], v[0::3, 1], v[0::3, 2] = cy, cp, cj
        v[1::3, 0], v[1::3, 1], v[1::3, 2] = Y[i], PHI[i], J[i]
        v[2::3, 0], v[2::3, 1], v[2::3, 2] = Y[j], PHI[j], J[j]
        v[:, 3], v[:, 4] = shell, self._z()
        v[:, 5:8] = col
        self.fills.append(v)

    # Tk-equivalent helpers -------------------------------------------------------------
    def glow(self, Y, PHI, J, shell, rgb, k, width, closed=False,
             layers=((5, 0.1), (2.4, 0.3), (1, 1.0)), base=(16, 13, 20), vis=0.0):
        for wm, bm in layers:
            col = lerp_rgb(base, rgb, min(1.0, bm * k))
            self.line(Y, PHI, J, shell, col01(col, 1.0 if k > 0.05 else k * 20),
                      max(1.0, width * wm) / 2, vis, closed)

    def trace(self, Y, PHI, J, shell, rgb, k, width, layers=((4, 0.12), (2, 0.35), (0.8, 1.0))):
        self.glow(Y, PHI, J, shell, rgb, k, width, layers=layers, vis=1.0)


class HeadGPU:
    def __init__(self, face):
        self.f = face
        F = type(face)
        self.PY, self.PW, self.PF, self.PB = F._PY, F._PW, F._PF, F._PB
        self.PW_F, self.PF_F, self.PB_F = F._PW_F, F._PF_F, F._PB_F
        self.mask_verts = self._build_mask()
        self.hair_verts = self._build_hair()
        face.scale = face.H * 0.27
        face.cam = 5.0
        self.frame = None

    # ---- static meshes (same topology and jitter as Face._build_mask / _build_hair)
    def _build_mask(self):
        f = self.f
        rng = np.random.default_rng(7)
        NR, NC = 17, 26
        y = np.linspace(0.97, -0.97, NR)[:, None] + np.zeros((NR, NC))
        phi = np.linspace(-np.pi, np.pi, NC, endpoint=False)[None, :] + np.zeros((NR, NC))
        jy = rng.uniform(-0.025, 0.025, (NR - 2, NC))
        jp = rng.uniform(-0.05, 0.05, (NR, NC))
        for c in range(NC):
            m = (NC - c) % NC
            if m == c:
                jp[:, c] = 0
            elif m < c:
                jy[:, c], jp[:, c] = jy[:, m], -jp[:, m]
        y[1:-1] += jy
        phi += jp
        idx = np.arange(NR * NC).reshape(NR, NC)
        tris = []
        for r in range(NR - 1):
            for c in range(NC):
                a, b = idx[r, c], idx[r, (c + 1) % NC]
                d, e = idx[r + 1, c], idx[r + 1, (c + 1) % NC]
                cm = (NC - c - 1) % NC
                diag = (r + c) % 2 if c <= cm else 1 - (r + cm) % 2
                tris += [(a, b, e), (a, e, d)] if diag else [(a, b, d), (b, e, d)]
        T = np.array(tris)
        V = f._surface(y, phi).reshape(-1, 3)
        cen = V[T].mean(1)
        eye_w = np.exp(-((np.abs(cen[:, 0]) - 0.27) / 0.16) ** 2 - ((cen[:, 1] - 0.15) / 0.12) ** 2) \
            * (cen[:, 2] > 0)
        Y, PH = y.ravel(), phi.ravel()
        v = np.zeros((len(T) * 3, 4), np.float32)
        v[:, 0], v[:, 1] = Y[T].ravel(), PH[T].ravel()
        v[:, 2] = np.repeat(eye_w, 3)
        return v

    def _build_hair(self):
        f = self.f
        rng = np.random.default_rng(5)
        NR, NC = 11, 30
        phi = np.linspace(-np.pi, np.pi, NC, endpoint=False) + np.pi / NC
        ap = np.abs(phi)
        ends = np.where(phi > 0, 0.46, 0.08 - 0.1 * np.clip((ap - 0.9) / 0.6, 0, 1))
        ends = np.where(ap > 2.2, np.minimum(ends, 0.0) + 0.1 * (phi > 0), ends)
        ends = ends - 0.07 * rng.random(NC)
        s_ = np.linspace(0, 1, NR)[:, None] ** 1.1
        HY = 1.0 - (1.0 - ends[None, :]) * s_
        HPHI = phi[None, :] - 0.22 * s_ + rng.uniform(-0.04, 0.04, (NR, NC))
        spike = rng.random((NR, NC)) ** 2 * (0.05 + 0.12 * (1 - s_)) * (np.arange(NR)[:, None] % 2 == 1)
        spike[0] = 0
        HPUFF = (1.0 + spike).ravel()
        HOFF = rng.uniform(0, 6, (NR, NC)).ravel()
        idx = np.arange(NR * NC).reshape(NR, NC)
        tris = []
        for r in range(NR - 1):
            for c in range(NC):
                a, b = idx[r, c], idx[r, (c + 1) % NC]
                d, e = idx[r + 1, c], idx[r + 1, (c + 1) % NC]
                tris += [(a, b, e), (a, e, d)] if (r + c) % 2 else [(a, b, d), (b, e, d)]
        T = np.array(tris)
        Y, PH = HY.ravel(), HPHI.ravel()
        yc, pc = Y[T].mean(1), PH[T].mean(1)
        keep = (np.abs(pc) > 0.92) | (yc > 0.6 + 0.22 * pc)
        T = T[keep]
        v = np.zeros((len(T) * 3, 4), np.float32)
        v[:, 0], v[:, 1] = Y[T].ravel(), PH[T].ravel()
        v[:, 2], v[:, 3] = HOFF[T].ravel(), HPUFF[T].ravel()
        return v

    # ---- per frame: replaces Face._draw_head
    def draw(self, state, t, prim, sec, k, openness):
        f = self.f
        W, H, u = f.W, f.H, f.u
        f.head_x = lerp(f.head_x, f.dodger.target_x - f.X, 0.035)
        f.hcx, f.hcy = f.head_x, H * 0.42 + math.sin(t * 0.6) * 4 * u
        lx, ly = f.look
        yaw = 0.16 * math.sin(t * 0.17) + lx * 0.18
        foc = f.dodger.focus
        if foc and state != "listening":
            fy_ = (foc[0] - f.X - f.head_x) / W
            fp_ = (foc[1] - H * 0.42) / H
            tgt = (max(-0.65, min(0.65, fy_ * 1.6)), max(-0.25, min(0.25, fp_ * 0.6)))
        else:
            tgt = (-(f.head_x - W / 2) / W * 0.9, 0.0)
        fo = getattr(f, "focus_turn", (0.0, 0.0))
        f.focus_turn = (lerp(fo[0], tgt[0], 0.05), lerp(fo[1], tgt[1], 0.05))
        yaw += f.focus_turn[0]
        if state == "listening":
            yaw *= 0.3
        pitch = 0.10 + 0.03 * math.sin(t * 0.23) - ly * 0.05 + f.focus_turn[1]
        f.rot = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch))
        if random.random() < 0.01:
            f.hcx += random.uniform(-14, 14) * u
        fem = f.fem
        eye_glow = 0.35 + 0.65 * max(0.0, openness) if state != "idle" else 0.25
        if state == "thinking":
            eye_glow *= 0.55 + 0.45 * abs(math.sin(t * 5))
        fx = getattr(f, "fem_x", 0.0)
        d = Deco()
        lvl = f.jaw if state == "speaking" else 0.0
        self._tron(d, state, t, prim, k, lvl)
        self._mouth(d, state, t, prim, k, lvl, fem)
        self._chip(d, state, t, prim, k, fem)
        if fx > 0.02:
            self._lips(d, prim, k * fx, lvl)
            self._undercut(d, state, t, prim, k * fx)
        self._eyes(d, state, prim, sec, k, openness, eye_glow, fx, lx)
        self.frame = dict(
            fem=fem, rot=f.rot, hc=(f.hcx, f.hcy), scale=f.scale, camd=f.cam, jaw=f.jaw * 0.6, t=t,
            thinking=int(state == "thinking"), eye_glow=eye_glow, prim=prim, sec=sec,
            kk=max(0.0, min(1.0, k)), hair_kk=max(0.0, min(1.0, k * fx)) if fx > 0.02 else 0.0,
            mask_w=max(1.0, 1.2 * u), hair_w=max(1.0, u),
            segs=d.finish(),
            fills=np.concatenate(d.fills) if d.fills else None)

    def profiles(self, fem):
        return (self.PY, lerp(self.PW, self.PW_F, fem), lerp(self.PF, self.PF_F, fem),
                lerp(self.PB, self.PB_F, fem))

    # ---- decoration ports (see the same-named parts of face_wallpaper.Face) ------------------
    def _tron(self, d, state, t, prim, k, lvl):
        f = self.f
        u, fem = f.u, f.fem
        if fem > 0.98:
            return
        k *= 1 - fem
        spin = t * (3.0 if state == "thinking" else 0.6)
        a = np.linspace(0, 2 * np.pi, 32)
        for side in (-1, 1):
            for rad, kk in ((1.0, 0.9), (0.6, 0.5)):
                d.trace(0.12 + 0.13 * rad * np.sin(a), side * (1.42 + 0.17 * rad * np.cos(a)), 0.0, 1.01,
                        prim, k * kk, 1.6 * u)
            for j in range(3):
                seg = a[:8] + spin * side + j * 2.1
                d.trace(0.12 + 0.105 * np.sin(seg), side * (1.42 + 0.137 * np.cos(seg)), 0.0, 1.012,
                        HOT if state != "idle" else prim, k * (0.6 + 0.4 * lvl), 2.2 * u,
                        layers=((3, 0.3), (1, 1.0)))

    def _mouth(self, d, state, t, prim, k, lvl, fem):
        f = self.f
        u = f.u
        n = 13
        ms = np.linspace(-0.19, 0.19, n)
        env = np.sin(np.linspace(0, np.pi, n)) ** 0.7
        wave = np.abs(np.sin(np.arange(n) * 1.3 + t * 22)) * 0.5 + 0.5
        gap = 0.004 + 0.022 * lvl * env * wave
        shell = 1.012 + 0.012 * fem
        Y = np.concatenate([-0.455 + gap * 0.3, -0.455 - gap])
        P = np.concatenate([ms, ms[::-1]])
        J = np.concatenate([np.zeros(n), np.full(n, f.jaw * 0.3)])
        if lvl > 0.04:
            d.glow(Y, P, J, shell, prim, k * min(1, 0.4 + lvl), 2.2 * u, closed=True,
                   layers=((7, 0.1), (3.5, 0.3), (1.4, 0.8)))
            d.fill(Y, P, J, shell, col01(lerp_rgb(prim, HOT, 0.35), k * min(1, 0.5 + lvl)))
        else:
            idle_k = 0.18 + (0.12 * (0.5 + 0.5 * math.sin(t * 2)) if state == "listening" else 0)
            d.glow(Y[:n], P[:n], 0.0, shell, prim, k * idle_k, 1.3 * u, layers=((3, 0.3), (1, 1.0)))

    def _chip(self, d, state, t, prim, k, fem):
        u = self.f.u
        cy, cp = 0.43, 0.0
        a = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
        for r, kk, w in ((1.0, 0.95, 1.8), (0.55, 0.6, 1.3)):
            d.glow(cy + 0.055 * r * np.sin(a), cp + 0.075 * r * np.cos(a), 0.0, 1.02, prim, k * kk, w * u,
                   layers=((5, 0.12), (2.4, 0.35), (1, 1.0)))
        rate = {"thinking": 6.0, "speaking": 3.0, "listening": 1.5}.get(state, 0.8)
        pulse = 0.5 + 0.5 * math.sin(t * rate)
        if state == "speaking":
            pulse = max(pulse * 0.5, self.f.jaw)
        d.glow([cy, cy], [cp - 0.012, cp + 0.012], 0.0, 1.025, lerp_rgb(prim, HOT, 0.6), k * (0.4 + 0.6 * pulse),
               4 * u, layers=((4, 0.2), (2, 0.5), (1, 1.0)))
        for path in (((cy + 0.01, cp + 0.075), (cy + 0.01, 0.30), (0.52, 0.42), (0.52, 0.62)),
                     ((cy - 0.02, cp + 0.07), (cy - 0.02, 0.26), (0.36, 0.36), (0.36, 0.58))):
            pts = densify(list(path), 6)
            Y, P = pts[:, 0], pts[:, 1]
            d.trace(Y, P, 0.0, 1.015, prim, k * 0.65, 1.2 * u, layers=((3, 0.15), (1, 1.0)))
            s = slide(Y, P, (t * rate * 0.15) % 1)
            if s is not None and state != "idle":
                d.trace(s[0], s[1], 0.0, 1.015, HOT, k * 0.7, 1.8 * u, layers=((3, 0.3), (1, 1.0)))
            r, w = 3 * u, max(1.0, 1.3 * u)
            d.dot(Y[-1], P[-1], 1.015, r + w / 2, col01(prim, k), 1.0)
            d.dot(Y[-1], P[-1], 1.015, max(0.5, r - w / 2), col01((16, 13, 20), 1), 1.0)

    def _lips(self, d, prim, k, lvl):
        u = self.f.u
        a = np.linspace(-1, 1, 15)
        bow = 0.012 * np.exp(-((np.abs(a) - 0.35) / 0.25) ** 2) - 0.008 * np.exp(-(a / 0.15) ** 2)
        for Y, P, J in ((-0.428 + bow - 0.02 * a ** 2, a * 0.2, 0.0),
                        (-0.478 - 0.028 * (1 - a ** 2) - 0.03 * lvl, a * 0.18, self.f.jaw * 0.3)):
            d.glow(Y, P, J, 1.03, prim, k * (0.45 + 0.4 * lvl), 1.6 * u, layers=((4, 0.12), (2, 0.3), (1, 1.0)))

    def _undercut(self, d, state, t, prim, k):
        u = self.f.u
        speed = {"thinking": 0.9, "speaking": 0.6, "listening": 0.3}.get(state, 0.0)
        for i, y0 in enumerate((0.36, 0.27, 0.18)):
            ph = np.linspace(1.0 + 0.08 * i, 2.15 - 0.1 * i, 24)
            Y = y0 - 0.04 * (ph - 1.0)
            d.trace(Y, ph, 0.0, 1.012, prim, k * 0.7, 1.4 * u, layers=((4, 0.1), (2, 0.3), (1, 1.0)))
            if speed:
                s = slide(Y, ph, (t * speed + i * 0.3) % 1)
                if s is not None:
                    d.trace(s[0], s[1], 0.0, 1.012, HOT, k * 0.7, 2 * u, layers=((3, 0.3), (1, 1.0)))

    def _eyes(self, d, state, prim, sec, k, openness, eye_glow, fx, lx):
        u = self.f.u
        e_open = max(0.0, openness) * (1.25 if state == "listening" else 1.0)
        if state == "idle":
            e_open *= 0.45
        for side in (-1, 1):
            th = 0.022 * e_open + 0.003
            ey = np.array([0.120, 0.150 + th, 0.188 + th * 0.4, 0.172 - th * 0.3, 0.130 - th * 0.6])
            ep = side * np.array([0.17, 0.33, 0.52, 0.40, 0.25])
            d.glow(ey, ep, 0.0, 1.015, sec, k * eye_glow, 3.2 * u, closed=True,
                   layers=((9, 0.08), (5, 0.18), (2.2, 0.55), (1, 1.0)))
            d.fill(ey, ep, 0.0, 1.015, col01(lerp_rgb(prim, HOT, 0.4), k * eye_glow))
            if e_open > 0.25:
                gp = side * (0.34 + lx * 0.07 * side)
                yy = 0.155 + 0.03 * (abs(gp) - 0.34) * 2
                d.glow([yy, yy], [gp - 0.05 * side, gp + 0.05 * side], 0.0, 1.02, HOT, k * eye_glow, 2.5 * u,
                       layers=((3, 0.35), (1, 1.0)))
            if fx > 0.02:
                for ly0, lp0, ly1, lp1 in ((0.186, 0.50, 0.225, 0.60), (0.180, 0.45, 0.228, 0.53),
                                           (0.172, 0.40, 0.215, 0.46)):
                    d.glow([ly0 + th * 0.4, ly1 + th * 0.4], [side * lp0, side * lp1], 0.0, 1.02, sec,
                           k * fx * eye_glow, 1.4 * u, layers=((3, 0.25), (1, 1.0)))


class HeadRenderer:
    """GL side: programs, static meshes, and the per-frame draw."""

    def __init__(self, ctx, W, H, head, seg_fs):
        import moderngl
        self.ctx, self.mgl, self.head = ctx, moderngl, head
        self.facet = ctx.program(vertex_shader=FACET_VS, geometry_shader=FACET_GS, fragment_shader=FACET_FS)
        self.dseg = ctx.program(vertex_shader=DECO_SEG_VS, fragment_shader=seg_fs)
        self.dfill = ctx.program(vertex_shader=DECO_FILL_VS, fragment_shader=DECO_FILL_FS)
        for p in (self.facet, self.dseg, self.dfill):
            p["screen"].value = (W, H)
        self.mask_buf = ctx.buffer(head.mask_verts.tobytes())
        self.hair_buf = ctx.buffer(head.hair_verts.tobytes())
        self.mask_vao = ctx.vertex_array(self.facet, [(self.mask_buf, "2f 2f", "yp", "extra")])
        self.hair_vao = ctx.vertex_array(self.facet, [(self.hair_buf, "2f 2f", "yp", "extra")])
        self.seg_buf = ctx.buffer(reserve=1 << 18, dynamic=True)
        self.fill_buf = ctx.buffer(reserve=1 << 16, dynamic=True)
        self.seg_vao = self.fill_vao = None

    @staticmethod
    def _uni(prog, name, value):
        if name in prog:
            prog[name].value = value

    def _common(self, prog, fr, prof):
        PY, PW, PF, PB = prof
        for name, arr in (("PY", PY), ("PW", PW), ("PF", PF), ("PB", PB)):
            if name in prog:
                prog[name].value = tuple(float(v) for v in arr)
        self._uni(prog, "fem", float(fr["fem"]))
        self._uni(prog, "rot", tuple(float(v) for v in fr["rot"]))
        self._uni(prog, "hc", tuple(float(v) for v in fr["hc"]))
        self._uni(prog, "scale", float(fr["scale"]))
        self._uni(prog, "camd", float(fr["camd"]))

    def _ensure(self, name, nbytes):
        buf = getattr(self, name)
        if buf.size < nbytes:
            buf.orphan(max(nbytes, buf.size * 2))
        return buf

    def draw(self):
        fr = self.head.frame
        if not fr:
            return
        ctx, mgl = self.ctx, self.mgl
        prof = self.head.profiles(fr["fem"])
        p = self.facet
        self._common(p, fr, prof)
        for name in ("t", "eye_glow"):
            self._uni(p, name, float(fr[name]))
        self._uni(p, "jaw", float(fr["jaw"]))
        self._uni(p, "thinking", int(fr["thinking"]))
        self._uni(p, "prim", tuple(float(v) for v in fr["prim"]))
        self._uni(p, "sec", tuple(float(v) for v in fr["sec"]))
        ctx.enable(mgl.DEPTH_TEST)
        ctx.depth_func = "<"
        self._uni(p, "hair", 0)
        self._uni(p, "kk", float(fr["kk"]))
        self._uni(p, "edge_w", float(fr["mask_w"]))
        self.mask_vao.render(mgl.TRIANGLES)
        if fr["hair_kk"] > 0:
            self._uni(p, "hair", 1)
            self._uni(p, "kk", float(fr["hair_kk"]))
            self._uni(p, "edge_w", float(fr["hair_w"]))
            self.hair_vao.render(mgl.TRIANGLES)
        # decorations always sit on top of the facets (their own depth slice), in draw order
        if fr["fills"] is not None:
            data = fr["fills"]
            self._ensure("fill_buf", data.nbytes).write(data.tobytes())
            if self.fill_vao is None:
                self.fill_vao = ctx.vertex_array(self.dfill, [(self.fill_buf, "3f 2f 3f", "ypj", "shz", "col")])
            self._common(self.dfill, fr, prof)
            self.fill_vao.render(mgl.TRIANGLES, vertices=len(data))
        if fr["segs"] is not None:
            data = fr["segs"]
            self._ensure("seg_buf", data.nbytes).write(data.tobytes())
            if self.seg_vao is None:
                self.seg_vao = ctx.vertex_array(self.dseg, [(self.seg_buf, "3f 3f 4f 3f /i", "a", "b", "shv", "col")])
            self._common(self.dseg, fr, prof)
            ctx.enable_direct(0x809E)
            self.seg_vao.render(mgl.TRIANGLE_STRIP, vertices=4, instances=len(data))
            ctx.disable_direct(0x809E)
