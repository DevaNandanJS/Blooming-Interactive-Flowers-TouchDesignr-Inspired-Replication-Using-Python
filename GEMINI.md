# GEMINI.md - Instructional Context for Flowers-For-Her

## Project Overview
**Flowers-For-Her** is an interactive Python application that recreates a TouchDesigner-based "Interactive Blooming Flower" installation. It uses computer vision to track hand gestures via a webcam, allowing users to control the growth and blooming of 3D flowers in real-time. Flowers appear at each fingertip position with fire-orange petals and a glowing bloom post-processing effect.

### Key Technologies
- **Python 3.13+**: The primary programming environment.
- **MediaPipe Tasks API**: Handles hand landmark detection (2-hand tracking with palm openness).
- **ModernGL**: High-performance, GPU-accelerated 3D rendering with multi-pass pipeline.
- **OpenCV**: Camera capture, image preprocessing, and hand skeleton overlay.
- **GLSL Shaders**: External shader files for flower rendering, bloom post-processing, and compositing.
- **Threading**: Decoupled CV processing from the 3D render loop for maximum performance.

---

## Architecture

### 1. Computer Vision Pipeline (`hand_tracking.py`)
- Uses the **MediaPipe Tasks API** for hand landmarker detection.
- Tracks **2 hands simultaneously** with handedness classification.
- Calculates **palm openness** (0.0 to 1.0) based on fingertip-to-MCP distances.
- Returns **fingertip positions** and **finger direction vectors** for all 10 fingers.
- Returns **full landmark data** for hand skeleton drawing via OpenCV.
- Left hand openness → controls flower **growth** (stem length + size).
- Right hand openness → controls flower **bloom** (petal opening) + glow intensity.

### 2. Rendering Engine (`main.py`)
- Built on **ModernGL** and **moderngl_window**.
- **Multi-pass rendering pipeline**:
  1. Webcam background (fullscreen textured quad)
  2. 3D flowers rendered to off-screen FBO at fingertip positions
  3. Bloom post-processing (brightness extraction + Gaussian blur)
  4. Final composite (webcam + flowers + additive glow)
- Orthographic projection with aspect-ratio-aware coordinate mapping.
- Flowers consist of procedural stems, 6-petal heads, and center spheres.

### 3. Procedural Geometry (`geometry.py`)
- Bird-of-Paradise / Strelitzia-style elongated petals with curvature.
- Thin cylindrical stems with configurable length.
- Smooth-shaded sphere for flower center (pistil).
- All geometry uses indexed rendering with computed smooth normals.

### 4. Shader Pipeline (`shaders/`)
- `flower.vert` / `flower.frag`: Petal bloom deformation + fire-orange gradient.
- `quad.vert` / `quad.frag`: Fullscreen quad for webcam background.
- `blur.frag`: 13-tap separable Gaussian blur (2-pass: horizontal + vertical).
- `composite.frag`: Additive glow blending with tone mapping.

---

## Building and Running

### Prerequisites
Ensure you are using a Python environment with the following dependencies:
```bash
pip install opencv-python mediapipe numpy scipy moderngl moderngl-window PyOpenGL pyrr
```

### Running the Project
Execute the main script from the root directory:
```bash
python main.py
```

---

## Development Conventions

- **Performance**: Always keep Computer Vision (CV) operations in the background `CVWorker` thread to avoid blocking the `on_render` loop.
- **Smoothing**: Use Exponential Moving Average (EMA) in the `on_render` method to smooth out data coming from the CV thread.
- **Geometry**: Petals, stems, and centers are procedurally generated in `geometry.py` with indexed rendering.
- **MediaPipe Migration**: The project uses the **Tasks API** (v0.10.31+) for compatibility with Python 3.13; legacy `mp.solutions` is not used.
- **Shaders**: External GLSL files in the `shaders/` directory for maintainability.

---

## Important Files
- `main.py`: Entry point, multi-pass rendering pipeline, flower instancing.
- `hand_tracking.py`: MediaPipe Tasks API integration for 2-hand tracking.
- `geometry.py`: Procedural mesh generation (petals, stems, centers).
- `shaders/`: GLSL shader source files.
- `hand_landmarker.task`: MediaPipe model file (binary).
- `refererence_files/InteractiveFlower/`: Original TouchDesigner project files (archival).
