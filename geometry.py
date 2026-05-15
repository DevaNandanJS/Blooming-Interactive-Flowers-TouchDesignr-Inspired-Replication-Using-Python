"""
Procedural Geometry Generation
Creates lily petal, stem, and center meshes for the interactive flower.
All meshes return interleaved (position, normal) float32 arrays + index arrays.
"""

import numpy as np
import math


def create_petal_mesh(length=0.75, max_width=0.20, segs_len=14, segs_wid=6):
    """
    Generate a lily-shaped petal mesh — wider and more rounded than BOP,
    with a center groove/ridge characteristic of lily tepals.
    Returns (vertex_data, index_data) where vertex_data is interleaved [x,y,z, nx,ny,nz].
    """
    verts = []
    for i in range(segs_len + 1):
        t = i / segs_len  # 0..1 along petal length

        # Width profile: broader and more rounded (sin^0.4 = wider belly)
        width = max_width * math.sin(t * math.pi) ** 0.4
        # Softer tip taper (starts later, gentler slope)
        if t > 0.75:
            width *= ((1.0 - t) / 0.25) ** 0.6

        y = t * length
        # Gentle upward curve along the petal
        z_curve = t * t * 0.08 * length

        for j in range(segs_wid + 1):
            s = (j / segs_wid - 0.5) * 2.0  # -1..1 across width
            x = s * width
            # Center groove — slight ridge along the midline (lily characteristic)
            z_groove = abs(s) * 0.025 * length * t
            # Cross-width curvature (cup the petal slightly)
            z_cup = -(s * s) * 0.03 * length * t
            verts.append([x, y, z_curve + z_groove + z_cup])

    verts = np.array(verts, dtype='f4')

    # Build triangle indices
    indices = []
    for i in range(segs_len):
        for j in range(segs_wid):
            a = i * (segs_wid + 1) + j
            b = a + 1
            c = (i + 1) * (segs_wid + 1) + j
            d = c + 1
            indices.extend([a, c, b, b, c, d])
    indices = np.array(indices, dtype='i4')

    # Compute smooth normals
    norms = np.zeros_like(verts)
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        n = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(n)
        if ln > 1e-8:
            n /= ln
        norms[i0] += n
        norms[i1] += n
        norms[i2] += n
    for i in range(len(norms)):
        ln = np.linalg.norm(norms[i])
        if ln > 1e-8:
            norms[i] /= ln
        else:
            norms[i] = [0, 0, 1]

    # Interleave: [x, y, z, nx, ny, nz]
    vertex_data = np.zeros((len(verts), 6), dtype='f4')
    vertex_data[:, 0:3] = verts
    vertex_data[:, 3:6] = norms
    return vertex_data.flatten(), indices


def create_stem_mesh(radius=0.012, height=1.0, segments=6, rings=4):
    """Generate a thin cylindrical stem mesh."""
    verts = []
    norms = []

    for i in range(rings + 1):
        t = i / rings
        y = t * height
        for j in range(segments + 1):
            angle = (j / segments) * 2.0 * math.pi
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            verts.append([x, y, z])
            norms.append([math.cos(angle), 0.0, math.sin(angle)])

    verts = np.array(verts, dtype='f4')
    norms = np.array(norms, dtype='f4')

    indices = []
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + 1
            c = (i + 1) * (segments + 1) + j
            d = c + 1
            indices.extend([a, c, b, b, c, d])
    indices = np.array(indices, dtype='i4')

    vertex_data = np.zeros((len(verts), 6), dtype='f4')
    vertex_data[:, 0:3] = verts
    vertex_data[:, 3:6] = norms
    return vertex_data.flatten(), indices


def create_center_mesh(radius=0.04, segments=8, rings=6):
    """Generate a small sphere for the flower center (pistil)."""
    verts = []
    norms = []

    for i in range(rings + 1):
        phi = (i / rings) * math.pi
        for j in range(segments + 1):
            theta = (j / segments) * 2.0 * math.pi
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            n = np.array([x, y, z])
            ln = np.linalg.norm(n)
            if ln > 1e-8:
                n /= ln
            else:
                n = np.array([0, 1, 0])
            verts.append([x, y, z])
            norms.append(n.tolist())

    verts = np.array(verts, dtype='f4')
    norms = np.array(norms, dtype='f4')

    indices = []
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + 1
            c = (i + 1) * (segments + 1) + j
            d = c + 1
            indices.extend([a, c, b, b, c, d])
    indices = np.array(indices, dtype='i4')

    vertex_data = np.zeros((len(verts), 6), dtype='f4')
    vertex_data[:, 0:3] = verts
    vertex_data[:, 3:6] = norms
    return vertex_data.flatten(), indices
