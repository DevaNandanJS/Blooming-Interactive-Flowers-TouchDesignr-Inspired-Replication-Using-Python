#version 330

in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_texture;
uniform vec2 u_direction;  // (1/w, 0) for horizontal or (0, 1/h) for vertical
uniform float u_intensity;

void main() {
    // 13-tap Gaussian blur kernel
    float weights[7] = float[](
        0.1964825501511404,
        0.2969069646728344,
        0.09447039785044732,
        0.010381362401148057,
        0.0,   // padding
        0.0,
        0.0
    );

    // Better weights for a wider, softer glow
    float w[7] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216, 0.004, 0.001);

    vec3 result = texture(u_texture, v_uv).rgb * w[0];

    for (int i = 1; i < 7; i++) {
        vec2 offset = u_direction * float(i) * 2.0;
        result += texture(u_texture, v_uv + offset).rgb * w[i];
        result += texture(u_texture, v_uv - offset).rgb * w[i];
    }

    f_color = vec4(result * u_intensity, 1.0);
}
