#version 330

uniform mat4 u_projection;
uniform mat4 u_model;
uniform float u_bloom;
uniform int u_part;  // 0=stem, 1=petal, 2=center, 3=stamen

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal;
out vec3 v_pos;
out float v_t;       // parametric coordinate along petal length
out float v_local_x; // original x coord for spot generation

void main() {
    vec3 pos = in_position;

    if (u_part == 1) {
        // --- Lily petal bloom deformation ---
        float t = pos.y;  // 0 at base, ~1 at tip

        // Recurve: lilies curl backward past horizontal
        float open_angle = u_bloom * 2.6;
        // Strong backward curl at tips (signature lily recurve)
        float curl = u_bloom * u_bloom * 1.0 * t * t;

        // Apply opening rotation around X axis
        float ca = cos(-open_angle);
        float sa = sin(-open_angle);
        vec3 rotated;
        rotated.x = pos.x;
        rotated.y = pos.y * ca - pos.z * sa;
        rotated.z = pos.y * sa + pos.z * ca;

        // Add recurve curl at tips
        rotated.z += curl;

        pos = rotated;
        v_t = t;
        v_local_x = in_position.x;
    } else {
        v_t = 0.0;
        v_local_x = 0.0;
    }

    vec4 world_pos = u_model * vec4(pos, 1.0);
    gl_Position = u_projection * world_pos;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_pos = world_pos.xyz;
}
