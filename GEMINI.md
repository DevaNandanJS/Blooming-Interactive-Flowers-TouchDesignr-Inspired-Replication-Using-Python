# GEMINI.md - Flowers-For-Her Project Documentation

## Project Overview
**Flowers-For-Her** is a high-performance interactive Python application that recreates a TouchDesigner-based "Interactive Blooming Flower" installation. It leverages computer vision for real-time hand gesture tracking and GPU-accelerated 3D rendering to create an immersive experience where users control the growth and blooming of virtual flowers.

### Core Experience
- **Webcam Integration**: The user's live video feed serves as the background.
- **Interactive Growth**: Left hand openness (fist to open palm) controls the height and scale of the branching lily tree.
- **Interactive Bloom**: Right hand openness controls the petal opening, stamen visibility, and glow intensity.
- **Aesthetic**: Fire-orange to golden-yellow gradients with a heavy emissive bloom (glow) effect.

---

## Technical Stack & Libraries
The project is optimized for **Python 3.13+** and utilizes the following core libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| `moderngl` | 5.12.0 | Core GPU-accelerated rendering. |
| `moderngl-window` | 3.1.1 | Window management and resource loading. |
| `mediapipe` | 0.10.35 | Hand tracking via the modern **Tasks API**. |
| `opencv-python` | 4.13.0 | Webcam capture, image processing, and skeleton overlay. |
| `numpy` | 2.4.4 | Efficient geometry and landmark data processing. |
| `pyrr` | 0.10.3 | Matrix math for 3D transformations. |
| `scipy` | 1.17.1 | Helper for complex spatial computations. |

---

## File Structure

```text
C:\Flowers-For-Her\
├── main.py                 # Main entry point; rendering pipeline & logic.
├── hand_tracking.py        # MediaPipe Tasks API integration & gesture analysis.
├── geometry.py             # Procedural mesh generation (petals, stems, centers).
├── hand_landmarker.task    # MediaPipe binary model file.
├── implementation_plan.md  # Original design goals and roadmap.
├── GEMINI.md               # (This file) Instructional context and documentation.
└── shaders/                # GLSL source files for the render pipeline.
    ├── flower.vert         # 3D vertex shader (bloom deformation).
    ├── flower.frag         # Fragment shader (fire-orange gradient + lighting).
    ├── quad.vert           # Fullscreen quad vertex shader (background/post-fx).
    ├── quad.frag           # Passthrough texture fragment shader.
    ├── blur.frag           # Separable Gaussian blur for bloom effect.
    └── composite.frag      # Final additive blend and tone mapping.
```

---

## Code Logic & Architecture

### 1. Computer Vision Pipeline (`hand_tracking.py`)
- **CVWorker Thread**: Decouples webcam capture and MediaPipe processing from the render thread to maintain high FPS (60+).
- **Hand Landmarks**: Tracks 2 hands simultaneously, providing 21 landmarks per hand.
- **Palm Openness**: Calculated by comparing fingertip-to-MCP distances against a normalized hand size (wrist to middle-MCP). Remapped from `[0.3, 0.9]` to `[0.0, 1.0]`.
- **Skeleton Drawing**: OpenCV is used to overlay a stylized "warm blue" skeleton on the raw webcam frame before sending it to the GPU.

### 2. Rendering Pipeline (`main.py`)
The application uses a **Multi-Pass Framebuffer Object (FBO)** architecture:

1.  **Background Pass**: Renders the webcam texture (mirrored) as a fullscreen quad.
2.  **Flower Pass**: Renders the 3D branching tree into an off-screen FBO (`flower_fbo`).
    - Uses **Exponential Moving Average (EMA)** to smooth hand data and prevent jitter.
    - Implements a branching logic where the main trunk supports several side branches, each topped with a lily flower.
3.  **Bloom Pass (Post-Processing)**:
    - **Horizontal Blur**: Blurs the flower texture horizontally into a half-res FBO.
    - **Vertical Blur**: Blurs the result vertically into another half-res FBO.
4.  **Composite Pass**:
    - Combines the background, the original flower render (alpha blended), and the blurred glow (additive blend).

### 3. Procedural Geometry (`geometry.py`)
- **Lily Petals**: Generated as a parametric mesh with a center ridge (groove) and width curvature.
- **Instancing**: The tree is built by combining a main trunk (cylindrical stem) with branch stems and multi-layered flower heads (6 petals in 2 layers).
- **Stamens/Pistils**: Procedurally placed and scaled based on the bloom amount.

### 4. Shader Logic (`shaders/`)
- **Bloom Deformation**: `flower.vert` rotates petals outward and adds organic "curl" based on the `u_bloom` uniform.
- **Color Gradient**: `flower.frag` interpolates between deep fire-orange and bright golden-yellow based on the vertex Y-position (normalized).
- **Glow Intensity**: The brightness of the bloom effect scales dynamically with the right hand's openness.

---

## Interactivity Controls

| Hand Gesture | Action | Implementation |
|--------------|--------|----------------|
| **Left Palm Open** | Grow Tree | Increases `trunk_h` and `plant_s` scale. |
| **Left Fist** | Shrink Tree | Tree retracts to a small bud at the base. |
| **Right Palm Open** | Bloom Flower | Petals rotate open; stamens/pistil appear; glow intensifies. |
| **Right Fist** | Close Flower | Petals fold into a tight bud; glow fades. |

---

## Development Notes
- **Smoothing**: The `smooth` factor in `on_render` is framerate-aware. Jitter in hand tracking is mitigated by the EMA.
- **Coordinate Mapping**: MediaPipe uses `(0,0)` at top-left, while ModernGL uses `(-1,-1)` at bottom-left. `main.py` handles the necessary flips and aspect-ratio corrections.
- **Performance**: High-resolution blur is computationally expensive; hence, bloom is performed at half-resolution.
