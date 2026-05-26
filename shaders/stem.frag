#version 330

in vec3 v_worldPos;
in vec3 v_normal;

uniform float u_bloom; // 0 = bud, 1 = fully open

out vec4 fragColor;

void main() {
    // Deep, dark royal/navy blue base color (richer contrast)
    vec3 base_color = vec3(0.02, 0.08, 0.42); 

    // Soft wrapping light
    vec3 lightDir = normalize(vec3(0.2, 1.0, 0.5));
    vec3 n = normalize(v_normal);
    float ndl = dot(n, lightDir) * 0.5 + 0.5;

    // Emissive/diffuse combine - base brightness scales slightly with bloom
    float base_emissive = 0.5 + 0.5 * u_bloom;
    vec3 lit = base_color * (ndl * 0.3 + 0.7) * base_emissive;

    // 50% Brighter cyan-blue rim highlight (fresnel) scaling with bloom
    vec3 rim_color = vec3(0.35, 0.70, 1.00);
    vec3 viewDir = normalize(-v_worldPos);
    float fresnel = 1.0 - max(dot(n, viewDir), 0.0);
    fresnel = pow(fresnel, 3.0);
    
    // Highlight grows stronger as flower blooms
    float highlight_strength = 0.4 + 0.6 * u_bloom;
    lit += rim_color * fresnel * 0.9 * highlight_strength;

    // Emissive boost for the bloom pass to pick it up and glow
    fragColor = vec4(lit, 0.90);
}
