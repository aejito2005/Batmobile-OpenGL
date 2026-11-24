// Batmobile glTF viewer + platform + Bezier + classic Phong + interactive GUI
// OpenGL 3.3, GLFW, GLAD, GLM, tinygltf 2.x
// Keep only one .cpp with main() included in your Visual Studio project.
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <cstdint>
#include <cstddef>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

namespace fs = std::filesystem;

const char* vertexShaderSrc = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uModel, uView, uProjection;
out vec3 FragPos;
out vec3 Normal;
out vec2 UV;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    FragPos = world.xyz;
    Normal = mat3(transpose(inverse(uModel))) * aNormal;
    UV = aUV;
    gl_Position = uProjection * uView * world;
}
)GLSL";

const char* fragmentShaderSrc = R"GLSL(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 UV;
out vec4 FragColor;
uniform vec3 uViewPos;
uniform vec4 uBaseColor;
uniform sampler2D uTexture;
uniform bool uHasTexture;
uniform bool uUnlit; // trajectory color, independent of the three lights
uniform vec3 uMainPos;
uniform vec3 uBluePos;
uniform vec3 uBackPos;
uniform bool uMainEnabled;
uniform bool uBlueEnabled;
uniform bool uBackEnabled;

vec3 pointLight(vec3 position, vec3 color, float power, float range,
                vec3 N, vec3 V, vec3 albedo) {
    vec3 delta = position - FragPos;
    float dist = length(delta);
    vec3 L = delta / max(dist, 0.0001);
    float diffuse = max(dot(N, L), 0.0);
    // Classic Phong specular: reflect the incoming light around the normal.
    vec3 R = reflect(-L, N);
    // No specular contribution when the light is behind the surface.
    // Exponent 9 approximately preserves the highlight width of Blinn-Phong 36.
    float specular = diffuse > 0.0 ? pow(max(dot(R, V), 0.0), 9.0) : 0.0;
    // Localized lights: changing their positions makes a noticeable difference.
    float attenuation = 1.0 / (1.0 + pow(dist / range, 2.0));
    // Specular light is added separately, so black paint can still show highlights.
    return color * power * attenuation *
           (albedo * diffuse + vec3(0.34) * specular);
}
void main() {
    vec4 albedo = uBaseColor * (uHasTexture ? texture(uTexture, UV) : vec4(1.0));
    if (albedo.a < 0.05) discard;
    if (uUnlit) { FragColor = albedo; return; }
    vec3 N = normalize(Normal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uViewPos - FragPos);
    // Base ambient lighting so the Tumbler remains visible.
    vec3 result = albedo.rgb * vec3(0.38, 0.42, 0.50);
    if (uMainEnabled)
        result += pointLight(uMainPos, vec3(1.0, 0.96, 0.87), 2.7, 2.2, N, V, albedo.rgb);
    if (uBlueEnabled)
        result += pointLight(uBluePos, vec3(0.16, 0.48, 1.0), 3.0, 2.5, N, V, albedo.rgb);
    if (uBackEnabled)
        result += pointLight(uBackPos, vec3(1.0, 0.57, 0.15), 2.8, 2.0, N, V, albedo.rgb);
    // Simple display tone mapping to avoid clipping intense highlights.
    result = result / (result + vec3(1.0));
    result = pow(result, vec3(1.0 / 2.2));
    FragColor = vec4(result, albedo.a);
}
)GLSL";

GLuint compileShader(GLenum type, const char* source) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]{};
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::cerr << "Shader error: " << log << '\n';
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint createProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    if (!vs || !fs) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096]{};
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::cerr << "Program error: " << log << '\n';
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

struct Vertex { glm::vec3 pos, normal; glm::vec2 uv; };
struct DrawPart {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei count = 0;
    GLenum mode = GL_TRIANGLES;
    int material = -1;
    glm::mat4 nodeTransform{ 1.0f };
};

struct Viewer {
    tinygltf::Model gltf;
    std::vector<DrawPart> parts;
    std::vector<GLuint> images;
    glm::vec3 center{ 0.0f };
    glm::vec3 normalizedSize{ 0.0f };
    float scale = 1.0f;
    bool loaded = false;

    static const unsigned char* accessorData(const tinygltf::Model& m, const tinygltf::Accessor& a) {
        if (a.bufferView < 0 || a.bufferView >= static_cast<int>(m.bufferViews.size())) return nullptr;
        const auto& view = m.bufferViews[a.bufferView];
        if (view.buffer < 0 || view.buffer >= static_cast<int>(m.buffers.size())) return nullptr;
        const auto& b = m.buffers[view.buffer];
        size_t offset = static_cast<size_t>(view.byteOffset) + static_cast<size_t>(a.byteOffset);
        return offset < b.data.size() ? b.data.data() + offset : nullptr;
    }

    static size_t stride(const tinygltf::Model& m, const tinygltf::Accessor& a) {
        const auto& view = m.bufferViews[a.bufferView];
        int s = a.ByteStride(view);
        return s > 0 ? static_cast<size_t>(s) : 0;
    }

    static bool readFloats(const tinygltf::Model& m, int index, int components, std::vector<float>& out) {
        if (index < 0 || index >= static_cast<int>(m.accessors.size())) return false;
        const auto& a = m.accessors[index];
        if (a.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
            tinygltf::GetNumComponentsInType(a.type) != components || a.sparse.isSparse) return false;
        const unsigned char* data = accessorData(m, a);
        if (!data) return false;
        size_t step = stride(m, a);
        if (step < sizeof(float) * static_cast<size_t>(components)) return false;
        out.resize(static_cast<size_t>(a.count) * components);
        for (size_t i = 0; i < static_cast<size_t>(a.count); ++i)
            std::memcpy(out.data() + i * components, data + i * step, components * sizeof(float));
        return true;
    }

    static bool readIndices(const tinygltf::Model& m, int index, std::vector<uint32_t>& out) {
        if (index < 0 || index >= static_cast<int>(m.accessors.size())) return false;
        const auto& a = m.accessors[index];
        if (a.type != TINYGLTF_TYPE_SCALAR || a.sparse.isSparse) return false;
        const unsigned char* data = accessorData(m, a);
        if (!data) return false;
        size_t bytes = 0;
        if (a.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) bytes = 1;
        else if (a.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) bytes = 2;
        else if (a.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) bytes = 4;
        else return false;
        size_t step = stride(m, a);
        if (step < bytes) return false;
        out.resize(static_cast<size_t>(a.count));
        for (size_t i = 0; i < out.size(); ++i) {
            uint32_t v = 0;
            std::memcpy(&v, data + i * step, bytes);
            out[i] = v;
        }
        return true;
    }

    static glm::mat4 transform(const tinygltf::Node& node) {
        if (node.matrix.size() == 16) {
            glm::mat4 result(1.0f);
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row)
                    result[col][row] = static_cast<float>(node.matrix[col * 4 + row]);
            return result;
        }
        glm::mat4 result(1.0f);
        if (node.translation.size() == 3)
            result = glm::translate(result, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
        if (node.rotation.size() == 4) {
            glm::quat q(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
            result *= glm::mat4_cast(q);
        }
        if (node.scale.size() == 3)
            result = glm::scale(result, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
        return result;
    }

    void addPrimitive(const tinygltf::Primitive& prim, const glm::mat4& world,
        glm::vec3& minBound, glm::vec3& maxBound) {
        auto it = prim.attributes.find("POSITION");
        if (it == prim.attributes.end()) return;
        std::vector<float> positions, normals, uv;
        if (!readFloats(gltf, it->second, 3, positions)) {
            std::cerr << "Unsupported POSITION accessor; skipping primitive\n";
            return;
        }
        size_t count = positions.size() / 3;
        auto nIt = prim.attributes.find("NORMAL");
        bool hasNormals = nIt != prim.attributes.end() && readFloats(gltf, nIt->second, 3, normals)
            && normals.size() == positions.size();
        auto uvIt = prim.attributes.find("TEXCOORD_0");
        bool hasUV = uvIt != prim.attributes.end() && readFloats(gltf, uvIt->second, 2, uv)
            && uv.size() == count * 2;
        std::vector<Vertex> vertices(count);
        for (size_t i = 0; i < count; ++i) {
            vertices[i].pos = glm::vec3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
            vertices[i].normal = hasNormals ? glm::vec3(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]) : glm::vec3(0.0f);
            vertices[i].uv = hasUV ? glm::vec2(uv[2 * i], uv[2 * i + 1]) : glm::vec2(0.0f);
            glm::vec3 p = glm::vec3(world * glm::vec4(vertices[i].pos, 1.0f));
            minBound = glm::min(minBound, p);
            maxBound = glm::max(maxBound, p);
        }
        std::vector<uint32_t> indices;
        if (prim.indices >= 0) {
            if (!readIndices(gltf, prim.indices, indices)) return;
        }
        else {
            indices.resize(count);
            for (size_t i = 0; i < count; ++i) indices[i] = static_cast<uint32_t>(i);
        }
        if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != TINYGLTF_MODE_TRIANGLE_STRIP &&
            prim.mode != TINYGLTF_MODE_TRIANGLE_FAN && prim.mode != -1) return;
        GLenum mode = prim.mode == TINYGLTF_MODE_TRIANGLE_STRIP ? GL_TRIANGLE_STRIP :
            prim.mode == TINYGLTF_MODE_TRIANGLE_FAN ? GL_TRIANGLE_FAN : GL_TRIANGLES;
        if (!hasNormals && mode == GL_TRIANGLES) {
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
                if (a >= count || b >= count || c >= count) continue;
                glm::vec3 normal = glm::cross(vertices[b].pos - vertices[a].pos,
                    vertices[c].pos - vertices[a].pos);
                if (glm::length(normal) > 0.000001f) {
                    vertices[a].normal += normal;
                    vertices[b].normal += normal;
                    vertices[c].normal += normal;
                }
            }
            for (auto& v : vertices)
                v.normal = glm::length(v.normal) > 0.000001f ? glm::normalize(v.normal) : glm::vec3(0, 1, 0);
        }
        DrawPart part;
        part.count = static_cast<GLsizei>(indices.size());
        part.mode = mode;
        part.material = prim.material;
        part.nodeTransform = world;
        glGenVertexArrays(1, &part.vao);
        glGenBuffers(1, &part.vbo);
        glGenBuffers(1, &part.ebo);
        glBindVertexArray(part.vao);
        glBindBuffer(GL_ARRAY_BUFFER, part.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, part.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
        glBindVertexArray(0);
        parts.push_back(part);
    }

    void visitNode(int index, const glm::mat4& parent, glm::vec3& mn, glm::vec3& mx, int depth = 0) {
        if (depth > 64 || index < 0 || index >= static_cast<int>(gltf.nodes.size())) return;
        const auto& node = gltf.nodes[index];
        glm::mat4 world = parent * transform(node);
        if (node.mesh >= 0 && node.mesh < static_cast<int>(gltf.meshes.size())) {
            for (const auto& prim : gltf.meshes[node.mesh].primitives) addPrimitive(prim, world, mn, mx);
        }
        for (int child : node.children) visitNode(child, world, mn, mx, depth + 1);
    }

    bool load(const fs::path& path) {
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        bool ok = loader.LoadASCIIFromFile(&gltf, &err, &warn, path.string());
        if (!warn.empty()) std::cout << "glTF warning: " << warn << '\n';
        if (!err.empty()) std::cerr << "glTF error: " << err << '\n';
        if (!ok) return false;
        std::cout << "glTF loaded: " << gltf.meshes.size() << " meshes, "
            << gltf.materials.size() << " materials, " << gltf.images.size() << " images\n";
        images.resize(gltf.images.size(), 0);
        for (size_t i = 0; i < gltf.images.size(); ++i) {
            const auto& img = gltf.images[i];
            if (img.width <= 0 || img.height <= 0 || img.image.empty() || img.bits != 8) continue;
            GLenum format = img.component == 4 ? GL_RGBA : img.component == 3 ? GL_RGB :
                img.component == 2 ? GL_RG : img.component == 1 ? GL_RED : 0;
            if (!format) continue;
            glGenTextures(1, &images[i]);
            glBindTexture(GL_TEXTURE_2D, images[i]);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, format, img.width, img.height, 0,
                format, GL_UNSIGNED_BYTE, img.image.data());
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        }
        glm::vec3 mn(1e30f), mx(-1e30f);
        if (gltf.scenes.empty()) {
            // Fallback for a file without scenes: visit all nodes with no parent.
            std::vector<bool> child(gltf.nodes.size(), false);
            for (const auto& n : gltf.nodes)
                for (int c : n.children) if (c >= 0 && c < static_cast<int>(child.size())) child[c] = true;
            for (size_t i = 0; i < child.size(); ++i) if (!child[i]) visitNode(static_cast<int>(i), glm::mat4(1), mn, mx);
        }
        else {
            int sceneIndex = gltf.defaultScene >= 0 && gltf.defaultScene < static_cast<int>(gltf.scenes.size()) ? gltf.defaultScene : 0;
            for (int root : gltf.scenes[sceneIndex].nodes) visitNode(root, glm::mat4(1), mn, mx);
        }
        if (parts.empty()) { std::cerr << "No compatible mesh primitives found\n"; return false; }
        center = (mn + mx) * 0.5f;
        float maxDim = std::max({ mx.x - mn.x, mx.y - mn.y, mx.z - mn.z });
        scale = maxDim > 0.00001f ? 3.0f / maxDim : 1.0f;
        normalizedSize = (mx - mn) * scale;
        loaded = true;
        std::cout << "GPU draw parts: " << parts.size() << ", scale: " << scale << '\n';
        return true;
    }

    void draw(GLuint program, const glm::mat4& rotation) const {
        glm::mat4 normalization = glm::scale(glm::mat4(1.0f), glm::vec3(scale)) *
            glm::translate(glm::mat4(1.0f), -center);
        GLint locModel = glGetUniformLocation(program, "uModel");
        for (const auto& part : parts) {
            glm::mat4 model = rotation * normalization * part.nodeTransform;
            glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
            glm::vec4 color(1.0f);
            GLuint texture = 0;
            bool doubleSided = false;
            if (part.material >= 0 && part.material < static_cast<int>(gltf.materials.size())) {
                const auto& mat = gltf.materials[part.material];
                doubleSided = mat.doubleSided;
                const auto& pbr = mat.pbrMetallicRoughness;
                if (pbr.baseColorFactor.size() == 4)
                    color = glm::vec4(pbr.baseColorFactor[0], pbr.baseColorFactor[1],
                        pbr.baseColorFactor[2], pbr.baseColorFactor[3]);
                int t = pbr.baseColorTexture.index;
                if (t >= 0 && t < static_cast<int>(gltf.textures.size())) {
                    int source = gltf.textures[t].source;
                    if (source >= 0 && source < static_cast<int>(images.size())) texture = images[source];
                }
            }
            if (doubleSided) glDisable(GL_CULL_FACE); else glEnable(GL_CULL_FACE);
            glUniform4fv(glGetUniformLocation(program, "uBaseColor"), 1, glm::value_ptr(color));
            glUniform1i(glGetUniformLocation(program, "uHasTexture"), texture != 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(part.vao);
            glDrawElements(part.mode, part.count, GL_UNSIGNED_INT, nullptr);
        }
        glBindVertexArray(0);
    }

    void cleanup() {
        for (auto& p : parts) {
            glDeleteVertexArrays(1, &p.vao);
            glDeleteBuffers(1, &p.vbo);
            glDeleteBuffers(1, &p.ebo);
        }
        for (GLuint t : images) if (t) glDeleteTextures(1, &t);
    }
};

// =============================================================
// CIRCULAR DISPLAY PLATFORM
// Reuses the model shader and the Vertex layout: position, normal, UV.
// A unit cylinder is scaled and positioned for each platform tier.
// =============================================================
struct Platform {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei indexCount = 0;
    float radius = 2.0f;
    float surfaceY = -0.5f;

    void create(const glm::vec3& carSize) {
        // Cover the full horizontal bounding box even when the car rotates.
        radius = 0.5f * glm::length(glm::vec2(carSize.x, carSize.z)) + 0.30f;
        // The model is centered at its bounding box center. Its lowest point
        // is therefore at -carSize.y/2, independent of its rotation around Y.
        surfaceY = -0.5f * carSize.y - 0.003f;

        constexpr int segments = 128;
        constexpr float tau = 6.28318530718f;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        // Top and bottom need separate normals from the cylindrical wall.
        for (int cap = 0; cap < 2; ++cap) {
            float y = cap == 0 ? 0.5f : -0.5f;
            glm::vec3 n(0.0f, cap == 0 ? 1.0f : -1.0f, 0.0f);
            uint32_t start = static_cast<uint32_t>(vertices.size());
            vertices.push_back({ glm::vec3(0.0f, y, 0.0f), n, glm::vec2(0.5f) });
            for (int i = 0; i <= segments; ++i) {
                float a = tau * i / segments;
                float x = std::cos(a), z = std::sin(a);
                vertices.push_back({ glm::vec3(x, y, z), n,
                                    glm::vec2(0.5f + x * 0.5f, 0.5f + z * 0.5f) });
            }
            for (int i = 0; i < segments; ++i) {
                uint32_t curr = start + 1 + static_cast<uint32_t>(i);
                uint32_t next = curr + 1;
                // CCW winding seen from outside the cylinder.
                if (cap == 0) indices.insert(indices.end(), { start, next, curr });
                else indices.insert(indices.end(), { start, curr, next });
            }
        }

        uint32_t side = static_cast<uint32_t>(vertices.size());
        for (int i = 0; i <= segments; ++i) {
            float a = tau * i / segments;
            float x = std::cos(a), z = std::sin(a);
            float u = static_cast<float>(i) / segments;
            glm::vec3 n(x, 0.0f, z);
            vertices.push_back({ glm::vec3(x,  0.5f, z), n, glm::vec2(u, 1.0f) });
            vertices.push_back({ glm::vec3(x, -0.5f, z), n, glm::vec2(u, 0.0f) });
        }
        for (int i = 0; i < segments; ++i) {
            uint32_t a = side + 2 * static_cast<uint32_t>(i);
            uint32_t b = a + 1, c = a + 2, d = a + 3;
            indices.insert(indices.end(), { a, c, b, c, d, b });
        }

        indexCount = static_cast<GLsizei>(indices.size());
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex),
            vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t),
            indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
        glBindVertexArray(0);
    }

    void drawTier(GLuint program, float r, float top, float height,
        const glm::vec4& color) const {
        glm::mat4 model = glm::translate(glm::mat4(1.0f),
            glm::vec3(0.0f, top - height * 0.5f, 0.0f));
        model = glm::scale(model, glm::vec3(r, height, r));
        glUniformMatrix4fv(glGetUniformLocation(program, "uModel"),
            1, GL_FALSE, glm::value_ptr(model));
        glUniform4fv(glGetUniformLocation(program, "uBaseColor"), 1, glm::value_ptr(color));
        // The platform has a solid color, so it needs no texture file.
        glUniform1i(glGetUniformLocation(program, "uHasTexture"), GL_FALSE);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, nullptr);
    }

    void draw(GLuint program) const {
        glEnable(GL_CULL_FACE);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(vao);
        // Three stacked tiers, from the dark base to the gray top surface.
        // Small radial steps leave the blue-gray decorative ring visible.
        drawTier(program, radius * 1.06f, surfaceY - 0.12f, 0.14f,
            glm::vec4(0.025f, 0.030f, 0.040f, 1.0f));
        drawTier(program, radius * 1.035f, surfaceY - 0.085f, 0.035f,
            glm::vec4(0.075f, 0.13f, 0.20f, 1.0f));
        drawTier(program, radius, surfaceY, 0.085f,
            glm::vec4(0.045f, 0.050f, 0.060f, 1.0f));
        glBindVertexArray(0);
    }

    void cleanup() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
    }
};


// =============================================================
// CLOSED BEZIER TRAJECTORY: four cubic segments in the XZ plane.
// Each segment approximates one quarter of a circle. The endpoints
// and first derivatives match at the joins, including the last join.
// =============================================================
struct BezierPath {
    glm::vec3 control[4][4]{};
    GLuint vao = 0, vbo = 0;
    GLsizei ribbonCount = 0, controlCount = 0;
    float radius = 4.0f, y = -1.0f;
    static constexpr int samplesPerSegment = 100;

    // B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3
    glm::vec3 position(int segment, float t) const {
        const auto& p = control[segment];
        float u = 1.0f - t;
        return u * u * u * p[0] + 3.0f * u * u * t * p[1] +
            3.0f * u * t * t * p[2] + t * t * t * p[3];
    }

    // Analytic derivative: tangent used to steer the car.
    glm::vec3 tangent(int segment, float t) const {
        const auto& p = control[segment];
        float u = 1.0f - t;
        return 3.0f * u * u * (p[1] - p[0]) + 6.0f * u * t * (p[2] - p[1]) +
            3.0f * t * t * (p[3] - p[2]);
    }

    // A normalized lap parameter in [0,1] selects the segment and local t.
    void evaluate(float lap, glm::vec3& pos, glm::vec3& dir) const {
        float q = 4.0f * glm::clamp(lap, 0.0f, 1.0f);
        int segment = std::min(static_cast<int>(q), 3);
        float t = q - static_cast<float>(segment);
        pos = position(segment, t);
        dir = glm::normalize(tangent(segment, t));
    }

    void create(float r, float groundY) {
        radius = r;
        y = groundY + 0.012f;
        // k = 4/3 * tan(pi/8), standard cubic circle approximation.
        const float k = 0.55228474983f * r;
        auto P = [&](float x, float z) { return glm::vec3(x, y, z); };
        control[0][0] = P(r, 0); control[0][1] = P(r, k);
        control[0][2] = P(k, r); control[0][3] = P(0, r);
        control[1][0] = P(0, r); control[1][1] = P(-k, r);
        control[1][2] = P(-r, k); control[1][3] = P(-r, 0);
        control[2][0] = P(-r, 0); control[2][1] = P(-r, -k);
        control[2][2] = P(-k, -r); control[2][3] = P(0, -r);
        control[3][0] = P(0, -r); control[3][1] = P(k, -r);
        control[3][2] = P(r, -k); control[3][3] = P(r, 0);

        // Render a narrow flat ribbon instead of a GL line: reliably visible
        // even on drivers that only support 1-pixel-wide GL lines.
        std::vector<Vertex> vertices;
        auto append = [&](const glm::vec3& p) {
            vertices.push_back({ p, glm::vec3(0,1,0), glm::vec2(0) });
            };
        const float halfWidth = 0.038f;
        for (int segment = 0; segment < 4; ++segment) {
            for (int i = 0; i < samplesPerSegment; ++i) {
                float t0 = float(i) / samplesPerSegment;
                float t1 = float(i + 1) / samplesPerSegment;
                glm::vec3 p0 = position(segment, t0), p1 = position(segment, t1);
                glm::vec3 n0 = glm::normalize(glm::cross(glm::vec3(0, 1, 0), tangent(segment, t0)));
                glm::vec3 n1 = glm::normalize(glm::cross(glm::vec3(0, 1, 0), tangent(segment, t1)));
                glm::vec3 a = p0 - halfWidth * n0, b = p0 + halfWidth * n0;
                glm::vec3 c = p1 - halfWidth * n1, d = p1 + halfWidth * n1;
                append(a); append(c); append(b);
                append(b); append(c); append(d);
            }
        }
        ribbonCount = static_cast<GLsizei>(vertices.size());
        // Optional teaching overlay: all four control polygons and points.
        for (int segment = 0; segment < 4; ++segment)
            for (int i = 0; i < 4; ++i) {
                glm::vec3 p = control[segment][i]; p.y += 0.025f; append(p);
            }
        controlCount = 16;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
        glBindVertexArray(0);
    }

    void draw(GLuint program, bool showControls) const {
        glm::mat4 identity(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 1, GL_FALSE, glm::value_ptr(identity));
        GLint color = glGetUniformLocation(program, "uBaseColor");
        GLint unlit = glGetUniformLocation(program, "uUnlit");
        glUniform1i(unlit, GL_TRUE);
        glUniform1i(glGetUniformLocation(program, "uHasTexture"), GL_FALSE);
        glDisable(GL_CULL_FACE);
        glBindVertexArray(vao);
        glUniform4f(color, 0.05f, 0.78f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, ribbonCount);
        if (showControls) {
            glUniform4f(color, 1.0f, 0.58f, 0.13f, 1.0f);
            for (int segment = 0;segment < 4;++segment)
                glDrawArrays(GL_LINE_STRIP, ribbonCount + 4 * segment, 4);
            glPointSize(8.0f);
            glDrawArrays(GL_POINTS, ribbonCount, controlCount);
            glPointSize(1.0f);
        }
        glBindVertexArray(0);
        glUniform1i(unlit, GL_FALSE);
        glEnable(GL_CULL_FACE);
    }

    void cleanup() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
    }
};

// =============================================================
// GUI: an independent 2D OpenGL overlay with GLFW mouse input.
// No extra libraries, font files or texture assets are needed.
// Keyboard input remains independent of the mouse and panel.
// =============================================================
struct ControlPanel {
    struct UiVertex { float x, y, r, g, b; };
    using Color = glm::vec3;
    GLuint program = 0, vao = 0, vbo = 0;
    std::vector<UiVertex> vertices;
    static constexpr float panelWidth = 284.0f, panelHeight = 720.0f;
    float scale = 1.0f;
    double mouseX = 0, mouseY = 0;
    bool pressed = false, down = false, wasDown = false;
    int active = -1, nextId = 0;
    const Color ink{ 0.91f,0.95f,1.0f }, muted{ 0.56f,0.65f,0.76f };
    const Color accent{ 0.23f,0.68f,1.0f };

    bool create() {
        const char* vs = R"GLSL(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec3 aColor;
uniform vec2 uScreen;
out vec3 color;
void main(){
    gl_Position=vec4(2.0*aPos.x/uScreen.x-1.0,1.0-2.0*aPos.y/uScreen.y,0.0,1.0);
    color=aColor;
})GLSL";
        const char* fs = R"GLSL(#version 330 core
in vec3 color;
out vec4 FragColor;
void main(){FragColor=vec4(color,1.0);}
)GLSL";
        GLuint v = compileShader(GL_VERTEX_SHADER, vs), f = compileShader(GL_FRAGMENT_SHADER, fs);
        if (!v || !f) { if (v) glDeleteShader(v); if (f) glDeleteShader(f); return false; }
        program = glCreateProgram();
        glAttachShader(program, v); glAttachShader(program, f); glLinkProgram(program);
        glDeleteShader(v); glDeleteShader(f);
        GLint ok = 0; glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) { glDeleteProgram(program); program = 0; return false; }
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(UiVertex), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(UiVertex), (void*)offsetof(UiVertex, r));
        glBindVertexArray(0);
        return true;
    }

    void begin(GLFWwindow* window, int w, int h) {
        scale = std::min(1.0f, std::min(float(h) / panelHeight, float(w) * 0.45f / panelWidth));
        scale = std::max(scale, 0.001f);
        glfwGetCursorPos(window, &mouseX, &mouseY);
        mouseX /= scale; mouseY /= scale;
        down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        pressed = down && !wasDown;
        wasDown = down;
        if (!down) active = -1;
        nextId = 0; vertices.clear();
    }

    void rect(float x, float y, float w, float h, Color c) {
        if (w <= 0 || h <= 0) return;
        auto add = [&](float a, float b) {vertices.push_back({ a * scale,b * scale,c.r,c.g,c.b });};
        add(x, y); add(x + w, y); add(x + w, y + h);
        add(x, y); add(x + w, y + h); add(x, y + h);
    }

    // Compact 5x7 bitmap alphabet drawn as small colored rectangles.
    static const unsigned char* glyph(char c) {
        static const unsigned char font[37][7] = {
            { 14,17,17,31,17,17,17 },{ 30,17,17,30,17,17,30 }, // A B
            { 14,17,16,16,16,17,14 },{ 30,17,17,17,17,17,30 }, // C D
            { 31,16,16,30,16,16,31 },{ 31,16,16,30,16,16,16 }, // E F
            { 14,17,16,23,17,17,14 },{ 17,17,17,31,17,17,17 }, // G H
            { 31,4,4,4,4,4,31 },{ 7,2,2,2,18,18,12 }, // I J
            { 17,18,20,24,20,18,17 },{ 16,16,16,16,16,16,31 }, // K L
            { 17,27,21,21,17,17,17 },{ 17,25,21,19,17,17,17 }, // M N
            { 14,17,17,17,17,17,14 },{ 30,17,17,30,16,16,16 }, // O P
            { 14,17,17,17,21,18,13 },{ 30,17,17,30,20,18,17 }, // Q R
            { 15,16,16,14,1,1,30 },{ 31,4,4,4,4,4,4 }, // S T
            { 17,17,17,17,17,17,14 },{ 17,17,17,17,17,10,4 }, // U V
            { 17,17,17,21,21,21,10 },{ 17,17,10,4,10,17,17 }, // W X
            { 17,17,10,4,4,4,4 },{ 31,1,2,4,8,16,31 }, // Y Z
            { 14,17,19,21,25,17,14 },{ 4,12,4,4,4,4,14 }, // 0 1
            { 14,17,1,2,4,8,31 },{ 30,1,1,14,1,1,30 }, // 2 3
            { 2,6,10,18,31,2,2 },{ 31,16,16,30,1,1,30 }, // 4 5
            { 14,16,16,30,17,17,14 },{ 31,1,2,4,8,8,8 }, // 6 7
            { 14,17,17,14,17,17,14 },{ 14,17,17,15,1,1,14 }, // 8 9
            { 0,0,0,0,0,0,0 }
        };
        static const unsigned char dash[7] = { 0,0,0,31,0,0,0 };
        static const unsigned char plus[7] = { 0,4,4,31,4,4,0 };
        static const unsigned char colon[7] = { 0,4,4,0,4,4,0 };
        static const unsigned char slash[7] = { 1,2,2,4,8,8,16 };
        static const unsigned char percent[7] = { 17,2,4,8,17,0,0 };
        if (c >= 'A' && c <= 'Z') return font[c - 'A'];
        if (c >= '0' && c <= '9') return font[26 + c - '0'];
        if (c == '-') return dash; if (c == '+') return plus;
        if (c == ':') return colon; if (c == '/') return slash; if (c == '%') return percent;
        return font[36];
    }

    void text(float x, float y, const std::string& label, float pixel, Color c) {
        for (char ch : label) {
            const auto* g = glyph(ch);
            for (int r = 0;r < 7;++r) for (int k = 0;k < 5;++k)
                if (g[r] & (1 << (4 - k))) rect(x + k * pixel, y + r * pixel, pixel, pixel, c);
            x += 6 * pixel;
        }
    }

    // Mouse buttons use a press edge for toggles; movement buttons can be held.
    bool button(float x, float y, float w, float h, const std::string& label,
        bool selected = false, bool hold = false, bool enabled = true) {
        int id = nextId++;
        bool hover = enabled && mouseX >= x && mouseX < x + w && mouseY >= y && mouseY < y + h;
        if (pressed && hover) active = id;
        bool hit = hover && active == id && (hold ? down : pressed);
        Color fill = selected ? Color(0.09f, 0.29f, 0.43f) : Color(0.13f, 0.18f, 0.25f);
        if (hover) fill += Color(0.06f);
        if (!enabled) fill = Color(0.08f, 0.11f, 0.15f);
        rect(x, y, w, h, selected ? accent : Color(0.24f, 0.31f, 0.40f));
        rect(x + 1, y + 1, w - 2, h - 2, fill);
        const float p = std::min(1.55f, (w - 8) / (6.0f * float(label.size())));
        text(x + (w - 6 * p * label.size() + p) * 0.5f, y + (h - 7 * p) * 0.5f, label, p, enabled ? ink : muted);
        return hit;
    }

    void section(float y, const std::string& label) {
        text(14, y, label, 1.45f, accent);
        rect(14, y + 15, 256, 1, Color(0.20f, 0.27f, 0.36f));
    }

    void render(int w, int h) {
        if (!program || vertices.empty()) return;
        glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE);
        glUseProgram(program);
        glUniform2f(glGetUniformLocation(program, "uScreen"), float(w), float(h));
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(UiVertex), vertices.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE);
    }
    void cleanup() {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (program) glDeleteProgram(program);
    }
};

static bool key(GLFWwindow* w, int k) { return glfwGetKey(w, k) == GLFW_PRESS; }

int main() {
    if (!glfwInit()) { std::cerr << "GLFW initialization failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(1200, 800, "Batmobile Tumbler - Phong / Bezier / GUI", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD initialization failed\n";
        glfwDestroyWindow(window); glfwTerminate(); return 1;
    }
    GLuint program = createProgram();
    if (!program) { glfwDestroyWindow(window); glfwTerminate(); return 1; }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Visual Studio often runs from the project folder, but may use another working directory.
    fs::path modelPath;
    const fs::path relative = fs::path("models") / "batmobile" / "scene.gltf";
    for (fs::path base : {fs::current_path(), fs::current_path().parent_path(),
        fs::current_path().parent_path().parent_path()}) {
        fs::path candidate = base / relative;
        if (fs::exists(candidate)) { modelPath = candidate; break; }
    }
    if (modelPath.empty()) {
        std::cerr << "Could not find models/batmobile/scene.gltf. Working directory: "
            << fs::current_path() << '\n';
        glDeleteProgram(program); glfwDestroyWindow(window); glfwTerminate(); return 1;
    }
    Viewer viewer;
    if (!viewer.load(modelPath)) {
        std::cerr << "Failed to load Batmobile: " << modelPath << '\n';
        glDeleteProgram(program); glfwDestroyWindow(window); glfwTerminate(); return 1;
    }
    Platform platform;
    platform.create(viewer.normalizedSize);
    // The platform stays at the origin. Give the moving car enough space
    // to clear the platform even at the widest point of its bounding box.
    const float carRadius = 0.5f * glm::length(glm::vec2(viewer.normalizedSize.x, viewer.normalizedSize.z));
    const float groundY = platform.surfaceY - 0.26f;
    BezierPath path;
    path.create(platform.radius * 1.06f + carRadius + 0.45f, groundY);
    const float floorRadius = path.radius + carRadius + 0.65f;
    // Elevated camera fits the entire path. WASD stays horizontal.
    const glm::vec3 initialCamera(0.0f, floorRadius * 1.35f, floorRadius * 1.85f);
    glm::vec3 camera = initialCamera;
    glm::vec3 front = glm::normalize(glm::vec3(0.0f, groundY, 0.0f) - initialCamera);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 mainLight(1.4f, 2.0f, 2.6f); // offsets relative to the car
    const glm::vec3 blueLight(-2.2f, 1.4f, -0.5f);
    const glm::vec3 backLight(0.8f, 1.7f, -2.8f);
    bool mainOn = true, blueOn = true, backOn = true;
    bool prev1 = false, prev2 = false, prev3 = false, prevR = false;
    bool prevB = false, prevP = false, prevT = false, prevG = false, prevF = false, prevH = false, prevC = false;
    bool moving = false, showPath = true, showControls = false, parked = true;
    float lap = 0.0f, angle = 0.0f;
    constexpr float lapSeconds = 12.0f;
    // glTF does not require a universal vehicle-forward axis. Infer the
    // long horizontal axis, assume its positive direction, and allow F to
    // reverse it if this particular asset faces the opposite way.
    const float forwardYaw = viewer.normalizedSize.x > viewer.normalizedSize.z ? 90.0f : 0.0f;
    float facingAdjustment = 0.0f;
    std::cout << "B: start/restart one Bezier lap (12 s) | P: pause/resume\n"
        << "T: path | G: control points | F: reverse car facing | C: park on platform\n"
        << "H: reset camera | WASD/QE: camera | LEFT/RIGHT: adjust car orientation\n"
        << "J/L: white light X | UP/DOWN: light height | 1/2/3: lights | R: reset lights | ESC: exit\n";
    ControlPanel gui;
    const bool guiReady = gui.create();
    bool guiVisible = guiReady, prevM = false;
    if (!guiReady) std::cerr << "GUI unavailable; all keyboard controls remain active.\n";
    std::cout << "M: show/hide control panel | Mouse: click or hold buttons\n";
    float last = static_cast<float>(glfwGetTime());
    while (!glfwWindowShouldClose(window)) {
        float now = static_cast<float>(glfwGetTime());
        float dt = std::min(now - last, 0.05f);
        last = now;
        glfwPollEvents();
        if (key(window, GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window, true);
        float speed = 3.0f * dt;
        glm::vec3 horizontal = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
        glm::vec3 right = glm::normalize(glm::cross(horizontal, up));
        if (key(window, GLFW_KEY_W)) camera += speed * horizontal;
        if (key(window, GLFW_KEY_S)) camera -= speed * horizontal;
        if (key(window, GLFW_KEY_A)) camera -= speed * right;
        if (key(window, GLFW_KEY_D)) camera += speed * right;
        if (key(window, GLFW_KEY_E)) camera += speed * up;
        if (key(window, GLFW_KEY_Q)) camera -= speed * up;
        if (key(window, GLFW_KEY_LEFT)) angle -= 60.0f * dt;
        if (key(window, GLFW_KEY_RIGHT)) angle += 60.0f * dt;
        bool kb = key(window, GLFW_KEY_B), kp = key(window, GLFW_KEY_P);
        bool kt = key(window, GLFW_KEY_T), kg = key(window, GLFW_KEY_G);
        bool kf = key(window, GLFW_KEY_F), kh = key(window, GLFW_KEY_H), kc = key(window, GLFW_KEY_C);
        if (kb && !prevB) { lap = 0.0f; moving = true; parked = false; angle = 0.0f; }
        if (kp && !prevP && !parked && lap < 1.0f) moving = !moving;
        if (kt && !prevT) showPath = !showPath;
        if (kg && !prevG) showControls = !showControls;
        if (kf && !prevF) facingAdjustment = std::fmod(facingAdjustment + 180.0f, 360.0f);
        if (kh && !prevH) camera = initialCamera;
        if (kc && !prevC) { parked = true; moving = false; lap = 0.0f; angle = 0.0f; }
        prevB = kb; prevP = kp; prevT = kt; prevG = kg; prevF = kf; prevH = kh; prevC = kc;
        if (moving) {
            lap = std::min(1.0f, lap + dt / lapSeconds);
            if (lap >= 1.0f) {
                moving = false;
                std::cout << "Bezier lap complete: 360 degrees. Press B to repeat.\n";
            }
        }
        // J/L moves the WHITE light across the Batmobile, not the camera.
        if (key(window, GLFW_KEY_J)) mainLight.x -= 2.5f * dt;
        if (key(window, GLFW_KEY_L)) mainLight.x += 2.5f * dt;
        if (key(window, GLFW_KEY_UP)) mainLight.y += 2.5f * dt;
        if (key(window, GLFW_KEY_DOWN)) mainLight.y -= 2.5f * dt;
        // Toggle on key press (not every frame while the key is held).
        bool k1 = key(window, GLFW_KEY_1), k2 = key(window, GLFW_KEY_2);
        bool k3 = key(window, GLFW_KEY_3), kr = key(window, GLFW_KEY_R);
        if (k1 && !prev1) mainOn = !mainOn;
        if (k2 && !prev2) blueOn = !blueOn;
        if (k3 && !prev3) backOn = !backOn;
        if (kr && !prevR) {
            mainLight = glm::vec3(1.4f, 2.0f, 2.6f);
            mainOn = blueOn = backOn = true;
        }
        prev1 = k1; prev2 = k2; prev3 = k3; prevR = kr;

        // GUI and keyboard share exactly the same scene state. There is no
        // keyboard focus capture: all the original key checks above remain active.
        bool km = key(window, GLFW_KEY_M);
        if (km && !prevM && guiReady) guiVisible = !guiVisible;
        prevM = km;
        int windowW = 0, windowH = 0;
        glfwGetWindowSize(window, &windowW, &windowH);
        gui.begin(window, windowW, windowH);
        const bool panelThisFrame = guiVisible;
        if (panelThisFrame && windowW > 0 && windowH > 0) {
            using Color = ControlPanel::Color;
            gui.rect(0, 0, ControlPanel::panelWidth, windowH / gui.scale, Color(0.045f, 0.065f, 0.10f));
            gui.rect(282, 0, 2, windowH / gui.scale, Color(0.23f, 0.43f, 0.63f));
            gui.text(14, 14, "BATMOBILE", 2.75f, gui.ink);
            gui.text(14, 39, "TUMBLER / OPENGL 3.3", 1.4f, gui.muted);
            if (gui.button(240, 10, 30, 23, "M")) guiVisible = false;
            gui.section(64, "BEZIER 360");
            gui.text(14, 88, parked ? "EN PLATAFORMA" : (moving ? "EN MOVIMIENTO" : (lap >= 1.0f ? "VUELTA COMPLETA" : "EN PAUSA")), 1.45f, gui.ink);
            gui.text(216, 88, std::to_string(int(lap * 100.0f)) + "%", 1.45f, gui.ink);
            gui.rect(14, 104, 256, 6, Color(0.16f, 0.23f, 0.31f));
            gui.rect(14, 104, 256 * lap, 6, gui.accent);
            if (gui.button(14, 120, 124, 27, "INICIAR B", moving)) { lap = 0; moving = true; parked = false; angle = 0; }
            if (gui.button(146, 120, 124, 27, moving ? "PAUSAR P" : "SEGUIR P", false, false, !parked && lap < 1)) moving = !moving;
            if (gui.button(14, 153, 124, 27, "CURVA T", showPath)) showPath = !showPath;
            if (gui.button(146, 153, 124, 27, "PUNTOS G", showControls)) showControls = !showControls;
            if (gui.button(14, 186, 124, 27, "ESTACIONAR C")) { parked = true; moving = false; lap = 0; angle = 0; }
            if (gui.button(146, 186, 124, 27, "INVERTIR F")) facingAdjustment = std::fmod(facingAdjustment + 180.0f, 360.0f);

            gui.section(229, "ILUMINACION PHONG");
            if (gui.button(14, 254, 256, 27, mainOn ? "1 BLANCA: ON" : "1 BLANCA: OFF", mainOn)) mainOn = !mainOn;
            if (gui.button(14, 287, 256, 27, blueOn ? "2 AZUL: ON" : "2 AZUL: OFF", blueOn)) blueOn = !blueOn;
            if (gui.button(14, 320, 256, 27, backOn ? "3 NARANJA: ON" : "3 NARANJA: OFF", backOn)) backOn = !backOn;
            gui.text(14, 357, "MOVER LUZ BLANCA", 1.35f, gui.muted);
            if (gui.button(14, 374, 124, 26, "IZQ J", false, true)) mainLight.x -= 2.5f * dt;
            if (gui.button(146, 374, 124, 26, "DER L", false, true)) mainLight.x += 2.5f * dt;
            if (gui.button(14, 406, 124, 26, "SUBIR", false, true)) mainLight.y += 2.5f * dt;
            if (gui.button(146, 406, 124, 26, "BAJAR", false, true)) mainLight.y -= 2.5f * dt;
            if (gui.button(14, 438, 256, 26, "RESTABLECER LUCES R")) { mainLight = glm::vec3(1.4f, 2.0f, 2.6f); mainOn = blueOn = backOn = true; }

            gui.section(480, "CAMARA Y ROTACION");
            if (gui.button(14, 505, 124, 26, "AVANZAR W", false, true)) camera += speed * horizontal;
            if (gui.button(146, 505, 124, 26, "RETROCEDER S", false, true)) camera -= speed * horizontal;
            if (gui.button(14, 537, 124, 26, "IZQ A", false, true)) camera -= speed * right;
            if (gui.button(146, 537, 124, 26, "DER D", false, true)) camera += speed * right;
            if (gui.button(14, 569, 124, 26, "SUBIR E", false, true)) camera += speed * up;
            if (gui.button(146, 569, 124, 26, "BAJAR Q", false, true)) camera -= speed * up;
            if (gui.button(14, 601, 124, 26, "GIRAR IZQ", false, true)) angle -= 60.0f * dt;
            if (gui.button(146, 601, 124, 26, "GIRAR DER", false, true)) angle += 60.0f * dt;
            if (gui.button(14, 633, 256, 26, "REINICIAR CAMARA H")) camera = initialCamera;
            gui.text(14, 673, "M: OCULTAR / MOSTRAR", 1.4f, gui.muted);
            gui.text(14, 690, "TECLADO SIEMPRE ACTIVO", 1.35f, gui.muted);
        }
        else if (guiReady && windowW > 0 && windowH > 0) {
            // A small mouse-accessible button reopens the hidden panel.
            if (gui.button(10, 10, 110, 28, "MENU M")) guiVisible = true;
        }

        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0) continue;
        glViewport(0, 0, width, height);
        glClearColor(0.035f, 0.055f, 0.085f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        int sceneLeft = 0;
        if (panelThisFrame && windowW > 0 && windowH > 0) {
            // Mouse coordinates are logical window units; viewport uses framebuffer pixels.
            sceneLeft = std::min(width - 1, int(ControlPanel::panelWidth * gui.scale * width / windowW));
        }
        glViewport(sceneLeft, 0, width - sceneLeft, height);
        glUseProgram(program);
        glm::mat4 view = glm::lookAt(camera, camera + front, up);
        float sceneAspect = float(width - sceneLeft) / float(height);
        float fov = glm::radians(55.0f);
        // Preserve the original horizontal field of view when the panel is open.
        if (panelThisFrame) fov = 2.0f * std::atan(std::tan(fov * 0.5f) * float(width) / float(width - sceneLeft));
        glm::mat4 projection = glm::perspective(fov, sceneAspect, 0.05f, 200.0f);
        glm::vec3 carPosition(0.0f, 0.0f, 0.0f);
        float carYaw = angle;
        if (!parked) {
            glm::vec3 tangent;
            path.evaluate(lap, carPosition, tangent);
            // Match the wheels to the surrounding floor, not the ribbon.
            carPosition.y = groundY + 0.5f * viewer.normalizedSize.y + 0.003f;
            carYaw = glm::degrees(std::atan2(tangent.x, tangent.z)) - forwardYaw + facingAdjustment + angle;
        }
        glm::mat4 carTransform = glm::translate(glm::mat4(1.0f), carPosition);
        carTransform = glm::rotate(carTransform, glm::radians(carYaw), glm::vec3(0, 1, 0));
        // Keep all three lights near the car as it moves around the platform.
        glm::vec3 mainWorld = mainLight + carPosition;
        glm::vec3 blueWorld = blueLight + carPosition;
        glm::vec3 backWorld = backLight + carPosition;
        glUniformMatrix4fv(glGetUniformLocation(program, "uView"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(program, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(program, "uViewPos"), 1, glm::value_ptr(camera));
        glUniform3fv(glGetUniformLocation(program, "uMainPos"), 1, glm::value_ptr(mainWorld));
        glUniform3fv(glGetUniformLocation(program, "uBluePos"), 1, glm::value_ptr(blueWorld));
        glUniform3fv(glGetUniformLocation(program, "uBackPos"), 1, glm::value_ptr(backWorld));
        glUniform1i(glGetUniformLocation(program, "uMainEnabled"), mainOn ? 1 : 0);
        glUniform1i(glGetUniformLocation(program, "uBlueEnabled"), blueOn ? 1 : 0);
        glUniform1i(glGetUniformLocation(program, "uBackEnabled"), backOn ? 1 : 0);
        glUniform1i(glGetUniformLocation(program, "uTexture"), 0);
        glUniform1i(glGetUniformLocation(program, "uUnlit"), GL_FALSE);
        // Reuse the cylinder mesh for the surrounding dark floor.
        glEnable(GL_CULL_FACE);
        glBindVertexArray(platform.vao);
        platform.drawTier(program, floorRadius, groundY, 0.12f, glm::vec4(0.065f, 0.075f, 0.095f, 1.0f));
        glBindVertexArray(0);
        platform.draw(program);
        if (showPath) path.draw(program, showControls);
        viewer.draw(program, carTransform);
        // Draw the GUI after all 3D geometry, independently of scene lights.
        glViewport(0, 0, width, height);
        if (windowW > 0 && windowH > 0) gui.render(windowW, windowH);
        glfwSwapBuffers(window);
    }
    gui.cleanup();
    path.cleanup();
    platform.cleanup();
    viewer.cleanup();
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
