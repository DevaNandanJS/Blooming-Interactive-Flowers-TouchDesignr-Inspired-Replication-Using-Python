"""Procedural mesh helpers for the flower tree.

Every builder returns ``(vertices, indices)`` as NumPy arrays ready
for upload to ModernGL VBOs/IBOs.

Vertex format (per vertex, float32):
    position    3 floats   (x, y, z)
    normal      3 floats   (nx, ny, nz)
    uv          2 floats   (u, v)
    normY       1 float    normalised-Y  (0 = root, 1 = tip)
    layerIdx    1 float    0.0 = outer layer, 1.0 = inner layer
    petalAngle  1 float    radial angle of this petal around flower centre

Total: 11 floats = 44 bytes per vertex.
"""

import numpy as np

VERT_FLOATS = 11  # pos(3)+norm(3)+uv(2)+normY(1)+layer(1)+petalAngle(1)


# ── Single petal ─────────────────────────────────────────────────────
def build_petal_mesh(n_u: int = 12, n_v: int = 8):
    """Single petal in canonical orientation — tip at +Y, root at origin.

    Parameters
    ----------
    n_u : int
        Samples along petal length (u ∈ [0, 1]).
    n_v : int
        Samples across petal width (v ∈ [-1, 1]).

    Returns
    -------
    verts : ndarray, shape (n_u*n_v, VERT_FLOATS)
    indices : ndarray of int32
    """
    verts = []

    for iu in range(n_u):
        u = iu / (n_u - 1)
        for iv in range(n_v):
            v = (iv / (n_v - 1)) * 2.0 - 1.0          # [-1, 1]

            # Width envelope: widest at ~40 %, tapers at tip
            w = np.sin(np.pi * u) * (1.0 - 0.3 * u * u)
            x = v * w * 0.5                            # half-width

            y = u                                       # length along petal

            # Centre ridge / groove
            z = -0.06 * (1.0 - v * v)

            # -- Analytical normal from parametric surface derivatives --
            # P(u,v) = (v·w(u)·0.5,  u,  -0.06·(1-v²))
            dw_du = (np.pi * np.cos(np.pi * u) * (1.0 - 0.3 * u * u)
                     + np.sin(np.pi * u) * (-0.6 * u))

            # ∂P/∂u = (v·dw/du·0.5,  1,  0)
            du_vec = np.array([v * dw_du * 0.5, 1.0, 0.0])
            # ∂P/∂v = (w·0.5,  0,  0.12·v)
            dv_vec = np.array([w * 0.5, 0.0, 0.12 * v])

            # normal = cross(∂P/∂v, ∂P/∂u) — outward-facing
            n = np.cross(dv_vec, du_vec)
            ln = np.linalg.norm(n)
            if ln > 1e-8:
                n /= ln
            else:
                n = np.array([0.0, 0.0, 1.0])

            verts.append([x, y, z,
                          n[0], n[1], n[2],
                          u, (v + 1.0) * 0.5,       # uv
                          u,                          # normY
                          0.0,                        # layer  (overridden)
                          0.0])                       # petalAngle (overridden)

    verts = np.array(verts, dtype=np.float32)

    # Indices — quad grid → two triangles per quad
    indices = []
    for iu in range(n_u - 1):
        for iv in range(n_v - 1):
            a = iu * n_v + iv
            b = a + 1
            c = a + n_v
            d = c + 1
            indices += [a, c, b,  b, c, d]

    return verts, np.array(indices, dtype=np.int32)


# ── Full flower head ─────────────────────────────────────────────────
def build_flower_mesh(n_petals: int = 6, n_layers: int = 2,
                      n_u: int = 12, n_v: int = 8,
                      base_size: float = 1.0):
    """Flower head: *n_layers* concentric rings of *n_petals* each.

    Petals are stored in **canonical orientation** (tip along +Y).
    Per-petal radial angle is written into the ``petalAngle`` attribute
    so the vertex shader can rotate each petal outward.

    Layer 0 (outer): ``n_petals`` at ``i·(2π/n)`` offset, full length.
    Layer 1 (inner): ``n_petals`` at ``i·(2π/n) + π/n`` offset, 72 % length.
    """
    petal_v, petal_i = build_petal_mesh(n_u, n_v)
    n_petal_verts = len(petal_v)

    all_verts = []
    all_idx   = []
    offset    = 0

    for layer in range(n_layers):
        petal_len    = base_size * (1.0 if layer == 0 else 0.72)
        angle_offset = 0.0 if layer == 0 else (np.pi / n_petals)
        layer_val    = float(layer)

        for i in range(n_petals):
            angle = (i / n_petals) * 2.0 * np.pi + angle_offset

            pv = petal_v.copy()
            # Scale positions by petal length
            pv[:, 0] *= petal_len   # x
            pv[:, 1] *= petal_len   # y
            pv[:, 2] *= petal_len   # z
            # Write per-petal metadata
            pv[:, 9]  = layer_val   # layer index
            pv[:, 10] = angle       # petal radial angle

            all_verts.append(pv)
            all_idx.append(petal_i + offset)
            offset += n_petal_verts

    vertices = np.concatenate(all_verts, axis=0)
    indices  = np.concatenate(all_idx,  axis=0)
    return vertices, indices


# ── Stamens ──────────────────────────────────────────────────────────
def build_stamen_mesh(n_stamens: int = 8, base_size: float = 1.0):
    """Thin quad-strip filaments + anther placeholder at tip.

    Format-compatible with flower mesh (11 floats per vertex).
    Shader uses ``petalAngle`` to fan stamens radially.
    """
    verts   = []
    indices = []
    offset  = 0

    for i in range(n_stamens):
        angle = (i / n_stamens) * 2.0 * np.pi
        filament_len = base_size * 0.22
        hw = base_size * 0.008          # half-width of quad-strip

        # 4 verts: base-left, base-right, tip-left, tip-right
        for t in (0.0, 1.0):
            for s in (-1.0, 1.0):
                verts.append([
                    s * hw,                 # x
                    t * filament_len,       # y
                    0.0,                    # z
                    0.0, 0.0, 1.0,          # normal (+Z)
                    t, (s + 1.0) * 0.5,     # uv
                    t,                      # normY
                    0.0,                    # layer
                    angle,                  # petalAngle
                ])

        a, b, c, d = offset, offset + 1, offset + 2, offset + 3
        indices += [a, c, b,  b, c, d]
        offset += 4

    return (np.array(verts, dtype=np.float32),
            np.array(indices, dtype=np.int32))


# ── Stem cylinder ───────────────────────────────────────────────────
def build_stem_segment(length: float = 1.0, radius: float = 0.02,
                       n_sides: int = 6):
    """Simple cylinder along +Y, centred at origin.

    Format-compatible (11 floats per vertex).  ``petalAngle`` is unused
    (set to 0.0) but present so the same VAO layout works.
    """
    verts   = []
    indices = []

    for ring in range(2):
        y = ring * length
        for i in range(n_sides):
            a = (i / n_sides) * 2.0 * np.pi
            x  = np.cos(a) * radius
            z  = np.sin(a) * radius
            nx = np.cos(a)
            nz = np.sin(a)
            verts.append([
                x, y, z,
                nx, 0.0, nz,
                i / n_sides, float(ring),   # uv
                float(ring),                # normY
                0.0,                        # layer
                0.0,                        # petalAngle (unused)
            ])

    for i in range(n_sides):
        a = i
        b = (i + 1) % n_sides
        c = a + n_sides
        d = b + n_sides
        indices += [a, c, b,  b, c, d]

    return (np.array(verts, dtype=np.float32),
            np.array(indices, dtype=np.int32))
