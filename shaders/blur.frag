#version 330

in vec2 v_uv;

uniform sampler2D u_texture;
uniform vec2 u_direction;   // (1/width, 0) for H pass or (0, 1/height) for V pass

out vec4 fragColor;

void main() {
    // 9-tap separable Gaussian blur, sigma ≈ 2.5
    float w[5] = float[](
        0.2270270270,
        0.1945945946,
        0.1216216216,
        0.0540540541,
        0.0162162162
    );

    vec4 color = texture(u_texture, v_uv) * w[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * float(i) * 1.5;
        color += texture(u_texture, v_uv + offset) * w[i];
        color += texture(u_texture, v_uv - offset) * w[i];
    }

    fragColor = color;
}
