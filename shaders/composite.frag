#version 330

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;
uniform sampler2D u_glow;
uniform float u_glow_strength;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 glow = texture(u_glow, v_uv).rgb;

    // Additive glow blending
    vec3 result = scene + glow * u_glow_strength;

    // Subtle tone mapping to prevent blowout
    result = result / (result + vec3(1.0));
    // Gamma correction
    result = pow(result, vec3(1.0 / 2.2));

    f_color = vec4(result, 1.0);
}
