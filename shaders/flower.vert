#version 330

// ── Vertex attributes (11 floats) ──────────────────────────────────
in vec3  a_position;
in vec3  a_normal;
in vec2  a_uv;
in float a_normY;       // 0 = root, 1 = tip
in float a_layer;       // 0 = outer, 1 = inner
in float a_petalAngle;  // radial angle around flower centre

// ── Uniforms ────────────────────────────────────────────────────────
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_proj;
uniform float u_bloom;   // 0 = closed bud, 1 = fully open

// ── Outputs to fragment shader ──────────────────────────────────────
out vec3  v_worldPos;
out vec3  v_normal;
out vec2  v_uv;
out float v_normY;
out float v_layer;

const float PI = 3.14159265359;

void main() {
    vec3 pos  = a_position;
    vec3 norm = a_normal;

    // ── Bloom deformation ───────────────────────────────────────────
    // open_angle: -0.42π (closed, petals folded inward) → -0.08π (open)
    float open_angle = mix(-0.42 * PI, -0.08 * PI, u_bloom);

    // Tilt angle: how far to rotate petal from vertical (+Y) toward
    // horizontal.  Progressive bend along petal length so the base
    // stays anchored and the tip moves the most.
    float tilt = (PI * 0.5 + open_angle) * a_normY;

    float st = sin(tilt);
    float ct = cos(tilt);

    // Rotate position around X axis (tilt outward from stem)
    vec3 tilted = vec3(
        pos.x,
        pos.y * ct - pos.z * st,
        pos.y * st + pos.z * ct
    );

    // Rotate normal around X axis
    vec3 n_tilted = vec3(
        norm.x,
        norm.y * ct - norm.z * st,
        norm.y * st + norm.z * ct
    );

    // Tip curl — bends tip back toward stem when closed
    tilted.y -= a_normY * a_normY * mix(0.1, 0.0, u_bloom);

    // ── Rotate to radial position around Y axis ─────────────────────
    float ca = cos(a_petalAngle);
    float sa = sin(a_petalAngle);

    vec3 rotated = vec3(
        tilted.x * ca - tilted.z * sa,
        tilted.y,
        tilted.x * sa + tilted.z * ca
    );
    vec3 n_rotated = vec3(
        n_tilted.x * ca - n_tilted.z * sa,
        n_tilted.y,
        n_tilted.x * sa + n_tilted.z * ca
    );

    // ── World / clip transforms ─────────────────────────────────────
    vec4 worldPos = u_model * vec4(rotated, 1.0);
    v_worldPos = worldPos.xyz;
    v_normal   = normalize(mat3(u_model) * n_rotated);
    v_uv       = a_uv;
    v_normY    = a_normY;
    v_layer    = a_layer;

    gl_Position = u_proj * u_view * worldPos;
}
