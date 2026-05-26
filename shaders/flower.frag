#version 330

in vec3  v_worldPos;
in vec3  v_normal;
in vec2  v_uv;
in float v_normY;
in float v_layer;

uniform float u_bloom;

out vec4 fragColor;

void main() {
    // ── Fire-orange (base) → bright golden-yellow (tip) ─────────────
    vec3 fire_orange = vec3(0.71, 0.16, 0.02);   // #B52805
    vec3 golden_tip  = vec3(1.00, 0.78, 0.12);   // #FFC71F

    vec3 petal_color = mix(fire_orange, golden_tip, v_normY);

    // ── Emissive brightness scales with bloom ───────────────────────
    float emissive = 0.4 + 0.6 * u_bloom;
    vec3 lit = petal_color * emissive;

    // ── Hemisphere diffuse lighting (soft wrap) ─────────────────────
    vec3 lightDir = normalize(vec3(0.2, 1.0, 0.5));
    vec3 n = normalize(v_normal);
    float ndl = dot(n, lightDir) * 0.5 + 0.5;   // wrap [0,1]
    lit += petal_color * ndl * 0.3;

    // ── Fresnel rim for petal-edge translucency ─────────────────────
    vec3 viewDir = normalize(-v_worldPos);        // camera at origin
    float fresnel = 1.0 - max(dot(n, viewDir), 0.0);
    fresnel *= fresnel;
    lit += petal_color * fresnel * 0.15 * u_bloom;

    // ── Translucency: outer ≈ 0.72, inner ≈ 0.52 ───────────────────
    float alpha = mix(0.72, 0.52, v_layer);

    fragColor = vec4(lit, alpha);
}
