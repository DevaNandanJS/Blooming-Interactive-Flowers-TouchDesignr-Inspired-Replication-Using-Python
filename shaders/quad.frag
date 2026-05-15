#version 330

in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_texture;

void main() {
    f_color = texture(u_texture, v_uv);
}
