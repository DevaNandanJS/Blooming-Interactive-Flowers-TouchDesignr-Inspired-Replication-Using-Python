"""Flowers-For-Her — GPU-accelerated interactive lily tree.

Run:  python main.py

Controls
--------
- Left hand open/close  → tree growth (height + scale).
- Right hand open/close → flower bloom (petal opening + glow).
"""

import struct
import math
import time
from pathlib import Path

import cv2
import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window.conf import settings
from pyrr import Matrix44, Vector3, Quaternion

from geometry import (
    build_flower_mesh,
    build_stamen_mesh,
    build_stem_segment,
    VERT_FLOATS,
)
from hand_tracking import CVWorker


# ── Configuration ────────────────────────────────────────────────────
WINDOW_SIZE     = (1280, 720)
BG_COLOR        = (0.03, 0.02, 0.04, 1.0)

# Tree structure
TRUNK_BASE_LEN  = 3.0        # GL units (30% smaller overall)
BRANCH_SHRINK   = 0.65
MAX_DEPTH       = 3
MIN_BRANCH_LEN  = 0.05


# ── Shader loading ──────────────────────────────────────────────────
def _load_shader(name: str) -> str:
    return (Path(__file__).parent / "shaders" / name).read_text()


# ── Inline line shader (branch skeleton) ────────────────────────────
_LINE_VERT = """
#version 330
in vec3 a_position;
in vec4 a_color;
uniform mat4 u_vp;
out vec4 v_color;
void main() {
    gl_Position = u_vp * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

_LINE_FRAG = """
#version 330
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = v_color;
}
"""


# ── Recursive branching tree ────────────────────────────────────────
def _build_branches(growth: float):
    """Build branch segments and flower positions.

    Returns
    -------
    segments : list of (x1, y1, z1, x2, y2, z2, thickness)
    flowers  : list of (x, y, z, size)
    """
    segments = []
    flowers  = []
    trunk_len = TRUNK_BASE_LEN * growth

    def _branch(x, y, z, angle, length, thickness, depth):
        if depth > MAX_DEPTH or length < MIN_BRANCH_LEN:
            return
        ex = x + math.cos(angle) * length
        ey = y + math.sin(angle) * length
        ez = z

        segments.append((x, y, z, ex, ey, ez, thickness))

        if depth == MAX_DEPTH or length < 0.25:
            flower_size = 0.42 + thickness * 0.18
            flowers.append((ex, ey, ez, flower_size))
            return

        n_children = 3 if depth < 2 else 2
        spread     = math.pi * (0.32 if depth < 2 else 0.36)
        child_len  = length * BRANCH_SHRINK
        child_thick = thickness * 0.7

        for i in range(n_children):
            frac = (i / (n_children - 1)) - 0.5 if n_children > 1 else 0.0
            c_angle = angle + frac * spread * 2
            _branch(ex, ey, ez, c_angle, child_len, child_thick, depth + 1)

    # Main trunk (moved down to -1.5 to make room for taller tree, shifted left to -0.9)
    trunk_base = (-0.9, -1.5, 0.0)
    trunk_top  = (-0.9, -1.5 + trunk_len, 0.0)
    segments.append((*trunk_base, *trunk_top, 0.028))

    # Side branches at different heights along the trunk (scaled lengths & thicknesses, shifted left)
    branch_defs = [
        (0.55, math.pi / 2 - 0.65, 1.57, 0.020),
        (0.55, math.pi / 2 + 0.65, 1.57, 0.020),
        (0.72, math.pi / 2 - 0.45, 1.32, 0.017),
        (0.72, math.pi / 2 + 0.45, 1.32, 0.017),
        (0.88, math.pi / 2 - 0.25, 1.07, 0.013),
        (0.88, math.pi / 2 + 0.25, 1.07, 0.013),
    ]

    for frac, angle, rel_len, thick in branch_defs:
        by = -1.5 + trunk_len * frac
        if by <= -1.5 + trunk_len:
            _branch(-0.9, by, 0.0, angle, rel_len * growth, thick, 1)

    # Top of trunk also branches straight up
    _branch(trunk_top[0], trunk_top[1], 0.0,
            math.pi / 2, 0.95 * growth, 0.015, 2)

    return segments, flowers


# ── Main window class ────────────────────────────────────────────────
class FlowerWindow(mglw.WindowConfig):
    title       = "Flowers-For-Her"
    window_size = WINDOW_SIZE
    gl_version  = (3, 3)
    aspect_ratio = None
    resizable   = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx: moderngl.Context

        # ── Hand tracking ────────────────────────────────────────────
        self.cv_worker = CVWorker()
        self.cv_worker.start()

        self.growth_raw    = 0.0
        self.bloom_raw     = 0.0
        self.growth_smooth = 0.0
        self.bloom_smooth  = 0.0
        self._last_time    = time.monotonic()

        # ── Build geometry ───────────────────────────────────────────
        flower_verts, flower_idx = build_flower_mesh(
            n_petals=6, n_layers=2, n_u=12, n_v=8, base_size=1.0,
        )
        stamen_verts, stamen_idx = build_stamen_mesh(
            n_stamens=8, base_size=1.0,
        )

        self.flower_vbo = self.ctx.buffer(flower_verts.tobytes())
        self.flower_ibo = self.ctx.buffer(flower_idx.tobytes())
        self.stamen_vbo = self.ctx.buffer(stamen_verts.tobytes())
        self.stamen_ibo = self.ctx.buffer(stamen_idx.tobytes())

        # ── Compile shaders ──────────────────────────────────────────
        self.flower_prog = self.ctx.program(
            vertex_shader=_load_shader("flower.vert"),
            fragment_shader=_load_shader("flower.frag"),
        )
        self.stem_prog = self.ctx.program(
            vertex_shader=_load_shader("stem.vert"),
            fragment_shader=_load_shader("stem.frag"),
        )
        self.quad_prog = self.ctx.program(
            vertex_shader=_load_shader("quad.vert"),
            fragment_shader=_load_shader("quad.frag"),
        )
        self.blur_prog = self.ctx.program(
            vertex_shader=_load_shader("quad.vert"),
            fragment_shader=_load_shader("blur.frag"),
        )
        self.composite_prog = self.ctx.program(
            vertex_shader=_load_shader("quad.vert"),
            fragment_shader=_load_shader("composite.frag"),
        )

        # ── Flower VAO (11-float vertex format) ─────────────────────
        # Note: a_uv is optimised away by the GLSL compiler (v_uv is
        # declared but unused in flower.frag), so we skip those 8 bytes
        # with padding (8x) instead of binding to a missing attribute.
        vao_fmt   = '3f 3f 8x 1f 1f 1f'
        vao_attrs = ('a_position', 'a_normal',
                     'a_normY', 'a_layer', 'a_petalAngle')

        self.flower_vao = self.ctx.vertex_array(
            self.flower_prog,
            [(self.flower_vbo, vao_fmt, *vao_attrs)],
            index_buffer=self.flower_ibo,
        )
        self.stamen_vao = self.ctx.vertex_array(
            self.flower_prog,
            [(self.stamen_vbo, vao_fmt, *vao_attrs)],
            index_buffer=self.stamen_ibo,
        )

        # ── Build 3D stem cylinder geometry ─────────────────────────
        stem_verts, stem_idx = build_stem_segment(length=1.0, radius=1.0, n_sides=8)
        self.stem_vbo = self.ctx.buffer(stem_verts.tobytes())
        self.stem_ibo = self.ctx.buffer(stem_idx.tobytes())

        # Stem VAO: positions (3f), normals (3f), then skip the other 5 attributes (20x)
        self.stem_vao = self.ctx.vertex_array(
            self.stem_prog,
            [(self.stem_vbo, '3f 3f 20x', 'a_position', 'a_normal')],
            index_buffer=self.stem_ibo,
        )

        # ── Fullscreen quad ──────────────────────────────────────────
        quad_verts = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
             1,  1,  1, 1,
            -1, -1,  0, 0,
             1,  1,  1, 1,
            -1,  1,  0, 1,
        ], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad_verts.tobytes())

        self.quad_vao_bg = self.ctx.vertex_array(
            self.quad_prog,
            [(self.quad_vbo, '2f 2f', 'a_position', 'a_uv')],
        )
        self.quad_vao_blur = self.ctx.vertex_array(
            self.blur_prog,
            [(self.quad_vbo, '2f 2f', 'a_position', 'a_uv')],
        )
        self.quad_vao_comp = self.ctx.vertex_array(
            self.composite_prog,
            [(self.quad_vbo, '2f 2f', 'a_position', 'a_uv')],
        )

        # ── FBOs ─────────────────────────────────────────────────────
        w, h = self.wnd.buffer_size
        self._setup_fbos(w, h)

        # ── Webcam texture (placeholder size, resized on first frame)
        self.webcam_tex = self.ctx.texture((1280, 720), 3, dtype='f1')
        self.webcam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ── FBO management ───────────────────────────────────────────────

    def _setup_fbos(self, w: int, h: int):
        """Create / recreate framebuffers at the given resolution."""
        # Full-res flower FBO (RGBA16F for HDR bloom)
        self.flower_tex = self.ctx.texture((w, h), 4, dtype='f2')
        self.flower_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.flower_depth = self.ctx.depth_renderbuffer((w, h))
        self.flower_fbo = self.ctx.framebuffer(
            color_attachments=[self.flower_tex],
            depth_attachment=self.flower_depth,
        )

        # Half-res blur FBOs
        hw, hh = max(w // 2, 1), max(h // 2, 1)

        self.blur_h_tex = self.ctx.texture((hw, hh), 4, dtype='f2')
        self.blur_h_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_h_fbo = self.ctx.framebuffer(
            color_attachments=[self.blur_h_tex],
        )

        self.blur_v_tex = self.ctx.texture((hw, hh), 4, dtype='f2')
        self.blur_v_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_v_fbo = self.ctx.framebuffer(
            color_attachments=[self.blur_v_tex],
        )

    def on_resize(self, width: int, height: int):
        self._setup_fbos(width, height)

    # ── Per-frame render ─────────────────────────────────────────────

    def on_render(self, time_val: float, frame_time: float):
        now = time.monotonic()
        dt  = now - self._last_time
        self._last_time = now

        # ── 1. Consume hand-tracking data ────────────────────────────
        fd = self.cv_worker.latest()
        if fd is not None:
            h_img, w_img = fd.frame.shape[:2]

            # BGR → RGB for OpenGL, then flip vertically (GL origin = bottom-left)
            rgb_frame = cv2.cvtColor(fd.frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.flip(rgb_frame, 0)

            if self.webcam_tex.size != (w_img, h_img):
                self.webcam_tex.release()
                self.webcam_tex = self.ctx.texture(
                    (w_img, h_img), 3, dtype='f1',
                )
                self.webcam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.webcam_tex.write(rgb_frame.tobytes())

            # Extract hand data — labels are anatomical:
            #   "Left" = user's left hand → growth
            #   "Right" = user's right hand → bloom
            self.growth_raw = 0.0
            self.bloom_raw  = 0.0
            for hand in fd.hands:
                if hand.label == "Left":
                    self.growth_raw = hand.openness
                elif hand.label == "Right":
                    self.bloom_raw = hand.openness

        # ── 2. EMA smoothing ─────────────────────────────────────────
        smooth = 1.0 - pow(0.02, dt) if dt > 0 else 0.0
        self.growth_smooth += (self.growth_raw - self.growth_smooth) * smooth
        self.bloom_smooth  += (self.bloom_raw  - self.bloom_smooth)  * smooth

        growth = self.growth_smooth
        bloom  = self.bloom_smooth

        # ── 3. Build branch tree ─────────────────────────────────────
        segments, flower_positions = _build_branches(max(growth, 0.05))

        # ── 4. Projection / View (adjusted to frame the larger tree) ─
        w, h   = self.wnd.buffer_size
        aspect = w / h if h > 0 else 1.0
        proj = Matrix44.perspective_projection(
            45.0, aspect, 0.1, 100.0, dtype='f4',
        )
        view = Matrix44.look_at(
            eye    = Vector3([0.0, 0.6, 5.2], dtype='f4'),
            target = Vector3([0.0, 0.4, 0.0], dtype='f4'),
            up     = Vector3([0.0, 1.0, 0.0], dtype='f4'),
            dtype  = 'f4',
        )

        # ═════════════════════════════════════════════════════════════
        # PASS 1 — Render branches + flowers into flower_fbo
        # ═════════════════════════════════════════════════════════════
        self.flower_fbo.use()
        self.flower_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        )

        # ── Render 3D Stems (Cylinders) ──────────────────────────────
        self.stem_prog['u_view'].write(view.tobytes())
        self.stem_prog['u_proj'].write(proj.tobytes())
        self.stem_prog['u_bloom'].value = bloom

        for seg in segments:
            x1, y1, z1, x2, y2, z2, thick = seg
            
            p1 = np.array([x1, y1, z1], dtype=np.float32)
            p2 = np.array([x2, y2, z2], dtype=np.float32)
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-5:
                continue
            dir_norm = direction / length

            # Default cylinder axis is +Y: (0, 1, 0)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # Calculate rotation axis and angle
            axis = np.cross(up, dir_norm)
            axis_len = np.linalg.norm(axis)
            
            if axis_len < 1e-6:
                if dir_norm[1] < 0:
                    rot = Matrix44.from_x_rotation(np.pi, dtype='f4')
                else:
                    rot = Matrix44.identity(dtype='f4')
            else:
                axis = axis / axis_len
                angle = np.arccos(np.clip(np.dot(up, dir_norm), -1.0, 1.0))
                quat = Quaternion.from_axis_rotation(axis, angle)
                rot = Matrix44.from_quaternion(quat, dtype='f4')

            scale = Matrix44.from_scale(Vector3([thick, length, thick], dtype='f4'), dtype='f4')
            translation = Matrix44.from_translation(Vector3(p1, dtype='f4'), dtype='f4')
            
            model = scale @ rot @ translation
            self.stem_prog['u_model'].write(model.tobytes())
            self.stem_vao.render()

        # ── Flowers at terminal branches ─────────────────────────────
        # Set view / proj / bloom once (they don't change per flower)
        self.flower_prog['u_view'].write(view.tobytes())
        self.flower_prog['u_proj'].write(proj.tobytes())
        self.flower_prog['u_bloom'].value = bloom

        for fx, fy, fz, fsize in flower_positions:
            model = (
                Matrix44.from_scale(
                    Vector3([fsize, fsize, fsize], dtype='f4'), dtype='f4',
                )
                @ Matrix44.from_translation(
                    Vector3([fx, fy, fz], dtype='f4'), dtype='f4',
                )
            )
            self.flower_prog['u_model'].write(model.tobytes())
            self.flower_vao.render()

            # Stamens (visible when blooming)
            if bloom > 0.05:
                self.stamen_vao.render()

        self.ctx.disable(moderngl.DEPTH_TEST)

        # ═════════════════════════════════════════════════════════════
        # PASS 2 — Horizontal Gaussian blur → blur_h_fbo (half-res)
        # ═════════════════════════════════════════════════════════════
        self.blur_h_fbo.use()
        self.blur_h_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.flower_tex.use(0)
        self.blur_prog['u_texture'].value = 0
        hw, hh = self.blur_h_tex.size
        self.blur_prog['u_direction'].value = (1.0 / hw, 0.0)
        self.quad_vao_blur.render()

        # ═════════════════════════════════════════════════════════════
        # PASS 3 — Vertical Gaussian blur → blur_v_fbo (half-res)
        # ═════════════════════════════════════════════════════════════
        self.blur_v_fbo.use()
        self.blur_v_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.blur_h_tex.use(0)
        self.blur_prog['u_texture'].value = 0
        self.blur_prog['u_direction'].value = (0.0, 1.0 / hh)
        self.quad_vao_blur.render()

        # ═════════════════════════════════════════════════════════════
        # PASS 4 — Composite → screen
        # ═════════════════════════════════════════════════════════════
        self.ctx.screen.use()
        self.ctx.clear(*BG_COLOR)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.webcam_tex.use(0)
        self.flower_tex.use(1)
        self.blur_v_tex.use(2)

        self.composite_prog['u_background'].value    = 0
        self.composite_prog['u_flower'].value         = 1
        self.composite_prog['u_bloom_blur'].value     = 2
        self.composite_prog['u_bloom_strength'].value = 0.4 + 0.6 * bloom

        self.quad_vao_comp.render()

        self.ctx.disable(moderngl.BLEND)

    # ── Cleanup ──────────────────────────────────────────────────────

    def on_close(self):
        self.cv_worker.stop()


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == '__main__':
    settings.WINDOW['class'] = 'moderngl_window.context.pyglet.Window'
    FlowerWindow.run()
