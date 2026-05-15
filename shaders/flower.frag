#version 330

in vec3 v_normal;
in vec3 v_pos;
in float v_t;
in float v_local_x;

out vec4 f_color;

uniform float u_bloom;
uniform int u_part;  // 0=stem, 1=petal, 2=center, 3=stamen

// Simple hash for procedural spots
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    // Lighting
    vec3 light_dir = normalize(vec3(0.4, 0.7, 0.5));
    vec3 light_dir2 = normalize(vec3(-0.3, 0.5, -0.6));
    float diff = max(dot(v_normal, light_dir), 0.0);
    float diff2 = max(dot(v_normal, light_dir2), 0.0) * 0.4;
    float ambient = 0.25;
    float lighting = ambient + diff * 0.7 + diff2;

    vec3 color;

    if (u_part == 1) {
        // --- Lily petal: warm orange gradient with dark spots ---
        vec3 base_col = vec3(1.0, 0.68, 0.12);    // Light warm orange (base)
        vec3 mid_col  = vec3(1.0, 0.48, 0.06);     // Rich orange (middle)
        vec3 tip_col  = vec3(0.95, 0.35, 0.04);    // Deep orange (tips)

        float t = clamp(v_t, 0.0, 1.0);
        if (t < 0.5) {
            color = mix(base_col, mid_col, t * 2.0);
        } else {
            color = mix(mid_col, tip_col, (t - 0.5) * 2.0);
        }

        // Procedural dark spots (tiger lily speckles)
        vec2 spot_uv = vec2(v_local_x * 50.0, v_t * 35.0);
        float spot = hash21(floor(spot_uv));
        // Only show spots in the inner region of the petal (not base or very tip)
        if (spot > 0.72 && t > 0.15 && t < 0.80 &&
            fract(spot_uv.x) > 0.25 && fract(spot_uv.y) > 0.25) {
            color *= vec3(0.30, 0.10, 0.03);  // Dark brown/maroon spots
        }

        // Emissive boost — kept subtle so petal details remain visible
        float emissive = 0.15 + 0.3 * u_bloom;
        color *= lighting * (1.0 + emissive);

    } else if (u_part == 2) {
        // --- Center pistil: yellow-green ---
        color = vec3(0.55, 0.65, 0.15) * lighting;

    } else if (u_part == 3) {
        // --- Stamen/anther: dark brown/maroon ---
        color = vec3(0.30, 0.10, 0.03) * lighting * 1.3;

    } else {
        // --- Stem: light blue/lavender ---
        color = vec3(0.35, 0.45, 0.90) * lighting;
    }

    f_color = vec4(color, 1.0);
}
