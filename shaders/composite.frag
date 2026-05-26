#version 330

in vec2 v_uv;

uniform sampler2D u_background;    // webcam feed
uniform sampler2D u_flower;        // sharp flower render (RGBA)
uniform sampler2D u_bloom_blur;    // blurred glow texture
uniform float     u_bloom_strength;

out vec4 fragColor;

void main() {
    // Background layer (webcam)
    vec3 bg = texture(u_background, v_uv).rgb;

    // Sharp flower (alpha-blended over background)
    vec4 flower = texture(u_flower, v_uv);

    // Blurred bloom (additive glow)
    vec3 glow = texture(u_bloom_blur, v_uv).rgb;

    // ── Composite ───────────────────────────────────────────────────
    vec3 out_color = bg;

    // Alpha-blend flower on top of background
    out_color = mix(out_color, flower.rgb, flower.a);

    // Add bloom glow (additive — drives the orange aura)
    out_color += glow * u_bloom_strength;

    // Reinhard tone-map to prevent clipping
    out_color = out_color / (out_color + vec3(1.0));

    fragColor = vec4(out_color, 1.0);
}
