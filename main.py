"""
Interactive Blooming Flower — Python Recreation
================================================
Multi-pass rendering pipeline with webcam compositing, hand-gesture-controlled
flower growth and bloom, and post-processing glow.

Controls:
  - Left hand open palm  → flowers GROW (stem length + bud size)
  - Right hand open palm → flowers BLOOM (petals open) + glow intensifies
  - Stems grow in the direction each finger is pointing
  - Flowers appear at all 10 fingertip positions
"""

import cv2
import threading
import queue
import time
import math
import os
import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44, Vector3
from hand_tracking import HandTracker
from geometry import create_petal_mesh, create_stem_mesh, create_center_mesh


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_shader(name):
    """Load a shader source file from the shaders/ directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'shaders', name)
    with open(path, 'r') as f:
        return f.read()


def build_rotation_from_direction(dx, dy):
    """
    Build a 4×4 rotation matrix that orients the local +Y axis along the 2D
    screen-space direction (dx, dy).  Since MediaPipe y increases downward,
    we negate dy so "up on screen" becomes +Y in world space.
    """
    # Negate dy to flip from screen coords to OpenGL coords
    dy = -dy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return Matrix44.identity(dtype='f4')

    dx /= length
    dy /= length

    # Angle between (0,1) and (dx,dy) — the default stem direction is +Y
    angle = math.atan2(dx, dy)  # rotation around Z
    return Matrix44.from_z_rotation(-angle, dtype='f4')


# ---------------------------------------------------------------------------
# CV Worker Thread
# ---------------------------------------------------------------------------

class CVWorker(threading.Thread):
    """Runs webcam capture + MediaPipe hand detection on a background thread."""

    def __init__(self, data_queue):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.running = True
        self.tracker = HandTracker()
        self.cap = cv2.VideoCapture(0)
        # Try to set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)  # Mirror
                self.tracker.process_frame(frame)
                hand_data = self.tracker.get_hand_data()

                # Draw skeleton overlay onto the frame
                self.tracker.draw_skeleton(frame, hand_data)

                # Package frame + hand data together
                payload = {
                    'frame': frame,
                    'hand_data': hand_data,
                }

                # Always keep only the latest data
                try:
                    while not self.data_queue.empty():
                        self.data_queue.get_nowait()
                    self.data_queue.put(payload)
                except queue.Full:
                    pass

            except Exception as e:
                if self.running:
                    print(f"CV Worker Error: {e}")
                break

            time.sleep(0.005)

        self.cap.release()

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class FlowerApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Interactive Blooming Flower"
    window_size = (1280, 720)
    aspect_ratio = None  # Allow flexible aspect

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ---- State ----
        self.growth = 0.0        # 0..1 controlled by left hand openness
        self.bloom_amount = 0.0  # 0..1 controlled by right hand openness
        self.hand_data = {'hands': []}
        self.last_frame = None
        self.webcam_size = (1280, 720)

        # ---- CV Thread ----
        self.data_queue = queue.Queue(maxsize=2)
        self.cv_worker = CVWorker(self.data_queue)
        self.cv_worker.start()

        # ---- Build Shaders ----
        self._build_shaders()

        # ---- Build Geometry ----
        self._build_geometry()

        # ---- Build FBOs for post-processing ----
        self._build_fbos()

        # ---- Webcam Texture ----
        self.webcam_tex = self.ctx.texture(self.webcam_size, 3, dtype='f1')
        self.webcam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ---------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------

    def _build_shaders(self):
        """Compile all shader programs."""
        # Flower shader (3D geometry with bloom deformation)
        self.flower_prog = self.ctx.program(
            vertex_shader=load_shader('flower.vert'),
            fragment_shader=load_shader('flower.frag'),
        )

        # Fullscreen quad shader (webcam background + texture passes)
        quad_vert = load_shader('quad.vert')
        self.quad_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=load_shader('quad.frag'),
        )
        self.blur_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=load_shader('blur.frag'),
        )
        self.composite_prog = self.ctx.program(
            vertex_shader=quad_vert,
            fragment_shader=load_shader('composite.frag'),
        )

    def _build_geometry(self):
        """Create VAOs for petal, stem, and center meshes, plus a fullscreen quad."""
        # Petal (lily-shaped: wider, rounder, with center groove)
        petal_vdata, petal_idx = create_petal_mesh(length=0.75, max_width=0.20,
                                                    segs_len=14, segs_wid=6)
        petal_vbo = self.ctx.buffer(petal_vdata)
        petal_ibo = self.ctx.buffer(petal_idx)
        self.petal_vao = self.ctx.vertex_array(
            self.flower_prog,
            [(petal_vbo, '3f 3f', 'in_position', 'in_normal')],
            petal_ibo,
        )

        # Stem
        stem_vdata, stem_idx = create_stem_mesh(radius=0.010, height=1.0)
        stem_vbo = self.ctx.buffer(stem_vdata)
        stem_ibo = self.ctx.buffer(stem_idx)
        self.stem_vao = self.ctx.vertex_array(
            self.flower_prog,
            [(stem_vbo, '3f 3f', 'in_position', 'in_normal')],
            stem_ibo,
        )

        # Flower center
        center_vdata, center_idx = create_center_mesh(radius=0.035)
        center_vbo = self.ctx.buffer(center_vdata)
        center_ibo = self.ctx.buffer(center_idx)
        self.center_vao = self.ctx.vertex_array(
            self.flower_prog,
            [(center_vbo, '3f 3f', 'in_position', 'in_normal')],
            center_ibo,
        )

        # Fullscreen quad (for background / post-processing passes)
        #   positions: (-1,-1) to (1,1)   UVs: (0,0) to (1,1)
        quad_data = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
             1,  1,  1, 1,
            -1, -1,  0, 0,
             1,  1,  1, 1,
            -1,  1,  0, 1,
        ], dtype='f4')
        quad_vbo = self.ctx.buffer(quad_data)

        self.quad_vao_bg = self.ctx.vertex_array(
            self.quad_prog, [(quad_vbo, '2f 2f', 'in_position', 'in_texcoord')])
        self.quad_vao_blur = self.ctx.vertex_array(
            self.blur_prog, [(quad_vbo, '2f 2f', 'in_position', 'in_texcoord')])
        self.quad_vao_comp = self.ctx.vertex_array(
            self.composite_prog, [(quad_vbo, '2f 2f', 'in_position', 'in_texcoord')])

    def _build_fbos(self):
        """Create off-screen framebuffer objects for the bloom post-processing pipeline."""
        w, h = self.window_size

        # FBO for rendering flowers (RGBA so we can composite over webcam)
        self.flower_tex = self.ctx.texture((w, h), 4, dtype='f2')
        self.flower_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.flower_depth = self.ctx.depth_renderbuffer((w, h))
        self.flower_fbo = self.ctx.framebuffer(
            color_attachments=[self.flower_tex],
            depth_attachment=self.flower_depth,
        )

        # Half-res FBOs for blur (cheaper, wider glow)
        bw, bh = w // 2, h // 2
        self.blur_tex_h = self.ctx.texture((bw, bh), 4, dtype='f2')
        self.blur_tex_h.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_fbo_h = self.ctx.framebuffer(color_attachments=[self.blur_tex_h])

        self.blur_tex_v = self.ctx.texture((bw, bh), 4, dtype='f2')
        self.blur_tex_v.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_fbo_v = self.ctx.framebuffer(color_attachments=[self.blur_tex_v])

    # ---------------------------------------------------------------
    # Render loop
    # ---------------------------------------------------------------

    def on_render(self, time_val, frame_time):
        """Main render callback — multi-pass pipeline."""

        # 1. Consume latest data from CV thread
        self._update_from_cv()

        # 2. Compute growth and bloom from hand openness (with EMA smoothing)
        self._update_controls(frame_time)

        # --- Pass 1: Webcam background to default framebuffer ---
        self.ctx.screen.use()
        self.ctx.clear(0.02, 0.02, 0.05)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._render_webcam_background()

        # --- Pass 2: Render flowers to off-screen FBO ---
        self.flower_fbo.use()
        self.flower_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._render_flowers()

        # --- Pass 3: Bloom post-processing (blur the flower texture) ---
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._render_bloom_blur()

        # --- Pass 4: Composite everything to screen ---
        self.ctx.screen.use()
        self._render_composite()

    # ---------------------------------------------------------------
    # Per-frame updates
    # ---------------------------------------------------------------

    def _update_from_cv(self):
        """Pull the latest frame + hand data from the CV thread."""
        try:
            while not self.data_queue.empty():
                payload = self.data_queue.get_nowait()
                self.last_frame = payload['frame']
                self.hand_data = payload['hand_data']
                self.webcam_size = (self.last_frame.shape[1], self.last_frame.shape[0])
        except queue.Empty:
            pass

        # Upload webcam frame to GPU texture
        if self.last_frame is not None:
            # Draw Grow / Bloom labels on the frame
            display_frame = self.last_frame.copy()
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, f"Grow: {self.growth:.2f}",
                        (int(w * 0.38), int(h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Bloom: {self.bloom_amount:.2f}",
                        (int(w * 0.58), int(h * 0.62)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 80, 255), 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 0)  # Flip for OpenGL origin
            # Resize webcam texture if camera resolution changed
            if (self.webcam_tex.width, self.webcam_tex.height) != (frame_rgb.shape[1], frame_rgb.shape[0]):
                self.webcam_tex.release()
                self.webcam_tex = self.ctx.texture((frame_rgb.shape[1], frame_rgb.shape[0]), 3, dtype='f1')
                self.webcam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.webcam_tex.write(frame_rgb.tobytes())

    def _update_controls(self, frame_time):
        """Map hand openness → growth / bloom with EMA smoothing."""
        target_growth = 0.0
        target_bloom = 0.0

        for hand in self.hand_data.get('hands', []):
            label = hand.get('label', '')
            openness = hand.get('openness', 0.0)

            # NOTE: MediaPipe "Left" means the left hand in the *mirrored* image,
            # which is actually the user's right hand.  We've already mirrored the
            # frame in the CV thread, so 'Left' label → user's LEFT hand.
            if label == 'Left':
                target_growth = max(target_growth, openness)
            else:
                target_bloom = max(target_bloom, openness)

        # EMA smoothing
        smooth = min(1.0, (frame_time if frame_time > 0 else 0.016) * 3.0)
        self.growth += (target_growth - self.growth) * smooth
        self.bloom_amount += (target_bloom - self.bloom_amount) * smooth

    # ---------------------------------------------------------------
    # Render passes
    # ---------------------------------------------------------------

    def _render_webcam_background(self):
        """Pass 1: Draw the webcam frame as a fullscreen background quad."""
        self.webcam_tex.use(0)
        self.quad_prog['u_texture'].value = 0
        self.quad_vao_bg.render(moderngl.TRIANGLES)

    # Branch definitions: (height_frac, angle_degrees, branch_len, flower_scale)
    # height_frac: where along the MAIN TRUNK the branch splits off (0=base, 1=top)
    # Pattern matches reference: lower branches spread wide, upper ones more upright.
    # Alternating left/right creates a natural tree silhouette.
    BRANCHES = [
        (0.22, -68,  0.55, 0.80),   # Lowest-left  (very wide spread)
        (0.28,  62,  0.48, 0.78),   # Lowest-right (very wide spread)
        (0.42, -42,  0.58, 0.88),   # Mid-lower-left
        (0.48,  48,  0.52, 0.84),   # Mid-lower-right
        (0.60, -55,  0.50, 0.86),   # Mid-upper-left (wide)
        (0.68,  38,  0.54, 0.83),   # Mid-upper-right
        (0.80, -22,  0.42, 0.90),   # Upper-left (more upright)
        (0.88,  15,  0.38, 0.88),   # Upper-right (more upright)
        (1.00,   0,  0.00, 1.00),   # Crown (flower on trunk tip)
    ]

    def _render_flowers(self):
        """Pass 2: Render branching lily tree — main trunk with branches and flowers."""
        w, h = self.wnd.buffer_size
        aspect = w / h if h > 0 else 1.0

        proj = Matrix44.orthogonal_projection(
            -aspect, aspect, -1.0, 1.0, -10.0, 10.0, dtype='f4'
        )
        self.flower_prog['u_projection'].write(proj)

        # Fixed base position of the main trunk
        base_x = -0.1
        base_y = -0.85

        # Main trunk height (grows with left-hand openness)
        trunk_h = 0.10 + self.growth * 1.50

        # Overall plant scale
        plant_s = 0.40 + self.growth * 0.60

        # Flower bud scale
        bud_scale = 0.15 + self.growth * 0.50

        # ---- Render main trunk ----
        trunk_model = (
            Matrix44.from_translation([base_x, base_y, 0.0], dtype='f4') *
            Matrix44.from_scale([plant_s * 0.5, trunk_h, plant_s * 0.5], dtype='f4')
        )
        self.flower_prog['u_model'].write(trunk_model.astype('f4'))
        self.flower_prog['u_bloom'].value = self.bloom_amount
        self.flower_prog['u_part'].value = 0  # stem
        self.stem_vao.render(moderngl.TRIANGLES)

        # ---- Render branches + lily flowers ----
        for (t_frac, angle_deg, branch_len, f_scale) in self.BRANCHES:
            self._render_branch_and_lily(
                base_x, base_y, trunk_h, plant_s,
                t_frac, angle_deg, branch_len,
                bud_scale * f_scale,
            )

    def _render_branch_and_lily(self, base_x, base_y, trunk_h, plant_s,
                                 t_frac, angle_deg, branch_len, flower_size):
        """Render one branch forking off the trunk, with a lily flower at the tip."""
        # Branch origin = point on the trunk at height t_frac
        branch_y = base_y + t_frac * trunk_h
        angle_rad = math.radians(angle_deg)

        # Effective branch length (grows with plant)
        eff_len = branch_len * (0.15 + self.growth * 0.85)

        # ---- Branch stem (only if there IS a branch, not the top flower) ----
        if branch_len > 0.01:
            br_model = (
                Matrix44.from_translation([base_x, branch_y, 0.0], dtype='f4') *
                Matrix44.from_z_rotation(angle_rad, dtype='f4') *
                Matrix44.from_scale([plant_s * 0.4, eff_len, plant_s * 0.4], dtype='f4')
            )
            self.flower_prog['u_model'].write(br_model.astype('f4'))
            self.flower_prog['u_bloom'].value = self.bloom_amount
            self.flower_prog['u_part'].value = 0
            self.stem_vao.render(moderngl.TRIANGLES)

        # Flower tip position
        tip_x = base_x + math.sin(angle_rad) * eff_len
        tip_y = branch_y + math.cos(angle_rad) * eff_len

        head = (
            Matrix44.from_translation([tip_x, tip_y, 0.0], dtype='f4') *
            Matrix44.from_scale([flower_size, flower_size, flower_size], dtype='f4')
        )

        # ---- Lily petals: 6 tepals in 2 alternating layers of 3 ----
        for p in range(6):
            petal_angle = (p / 6.0) * math.pi * 2.0
            layer = p % 2   # alternating inner/outer
            tilt = 0.03 + layer * 0.04   # inner layer slightly more upright
            scale = 1.0 - layer * 0.06   # outer layer slightly larger

            petal_model = (
                head *
                Matrix44.from_scale([scale, scale, scale], dtype='f4') *
                Matrix44.from_y_rotation(petal_angle, dtype='f4') *
                Matrix44.from_x_rotation(-tilt, dtype='f4')
            )
            self.flower_prog['u_model'].write(petal_model.astype('f4'))
            self.flower_prog['u_bloom'].value = self.bloom_amount
            self.flower_prog['u_part'].value = 1
            self.petal_vao.render(moderngl.TRIANGLES)

        # ---- Stamens: 6 thin rods with anthers (visible when blooming) ----
        if self.bloom_amount > 0.08:
            for s in range(6):
                stamen_angle = (s / 6.0) * math.pi * 2.0 + math.pi / 6.0
                stamen_tilt = 0.25 + self.bloom_amount * 0.55

                # Stamen rod (reuse stem VAO, very thin)
                stamen_model = (
                    head *
                    Matrix44.from_y_rotation(stamen_angle, dtype='f4') *
                    Matrix44.from_x_rotation(-stamen_tilt, dtype='f4') *
                    Matrix44.from_scale([0.12, 0.45, 0.12], dtype='f4')
                )
                self.flower_prog['u_model'].write(stamen_model.astype('f4'))
                self.flower_prog['u_part'].value = 3  # stamen color
                self.stem_vao.render(moderngl.TRIANGLES)

                # Anther (tiny sphere at stamen tip)
                anther_model = (
                    head *
                    Matrix44.from_y_rotation(stamen_angle, dtype='f4') *
                    Matrix44.from_x_rotation(-stamen_tilt, dtype='f4') *
                    Matrix44.from_translation([0.0, 0.45, 0.0], dtype='f4') *
                    Matrix44.from_scale([0.08, 0.12, 0.08], dtype='f4')
                )
                self.flower_prog['u_model'].write(anther_model.astype('f4'))
                self.flower_prog['u_part'].value = 3  # anther color
                self.center_vao.render(moderngl.TRIANGLES)

        # ---- Center pistil (visible when blooming) ----
        if self.bloom_amount > 0.10:
            self.flower_prog['u_model'].write(head.astype('f4'))
            self.flower_prog['u_part'].value = 2
            self.center_vao.render(moderngl.TRIANGLES)

    def _render_bloom_blur(self):
        """Pass 3: Two-pass Gaussian blur on the flower texture for glow."""
        # Horizontal blur: flower_tex → blur_fbo_h
        self.blur_fbo_h.use()
        self.blur_fbo_h.clear(0.0, 0.0, 0.0, 0.0)
        self.flower_tex.use(0)
        self.blur_prog['u_texture'].value = 0
        bw = self.blur_tex_h.width
        bh = self.blur_tex_h.height
        self.blur_prog['u_direction'].value = (1.0 / bw, 0.0)
        self.blur_prog['u_intensity'].value = 0.7 + self.bloom_amount * 0.8
        self.quad_vao_blur.render(moderngl.TRIANGLES)

        # Vertical blur: blur_tex_h → blur_fbo_v
        self.blur_fbo_v.use()
        self.blur_fbo_v.clear(0.0, 0.0, 0.0, 0.0)
        self.blur_tex_h.use(0)
        self.blur_prog['u_direction'].value = (0.0, 1.0 / bh)
        self.blur_prog['u_intensity'].value = 0.7 + self.bloom_amount * 0.8
        self.quad_vao_blur.render(moderngl.TRIANGLES)

    def _render_composite(self):
        """Pass 4: Composite flowers + glow on top of the webcam background."""
        # Draw flowers with alpha blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.flower_tex.use(0)
        self.quad_prog['u_texture'].value = 0
        self.quad_vao_bg.render(moderngl.TRIANGLES)

        # Draw glow with additive blending
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.blur_tex_v.use(0)
        self.quad_prog['u_texture'].value = 0
        self.quad_vao_bg.render(moderngl.TRIANGLES)

        # Reset blend mode
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    def close(self):
        self.cv_worker.stop()
        self.cv_worker.join(timeout=2.0)


if __name__ == '__main__':
    mglw.run_window_config(FlowerApp)
