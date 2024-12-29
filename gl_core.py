from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import logging
import ctypes
from memory_manager import MemoryPool
from typing import Optional, Tuple
import logging
from performance_core import (
    AsyncTextureManager,
    TextureUpdateTask,
    MemoryAlignment,
    BufferSpec,
    AlignedBuffer,
)
from memory_manager import MemoryPool, MemoryType, BufferSpec

logger = logging.getLogger(__name__)


class GLCore:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.texture_id = None

        self.vertex_pool = MemoryPool(
            block_sizes=[4 * 16, 4 * 6, width * height * 3 * 4], blocks_per_size=2
        )

        self.texture_manager = AsyncTextureManager(max_workers=2)
        self._setup_gl()

    def _setup_gl(self) -> None:
        try:
            self.vao = glGenVertexArrays(1)
            glBindVertexArray(self.vao)

            self._create_shader_program()

            self._setup_vertex_data()

            self._setup_texture()

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            error = glGetError()
            if error != GL_NO_ERROR:
                logger.error(f"OpenGL initialization error: {error}")

        except Exception as e:
            logger.error(f"Failed to setup OpenGL: {e}")
            raise

    def _setup_texture(self) -> None:
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB32F,
            self.width,
            self.height,
            0,
            GL_RGB,
            GL_FLOAT,
            None,
        )

        glGenerateMipmap(GL_TEXTURE_2D)

    def _create_shader_program(self) -> None:
        vertex_source = """
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        out vec2 v_texcoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
        """

        fragment_source = """
        #version 330 core
        uniform sampler2D tex;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec3 color = texture(tex, v_texcoord).rgb;
            float gray = dot(color, vec3(0.299, 0.587, 0.114));
            gray *= 0.75;
            gray = pow(gray, 1.4);
            
            vec2 center = v_texcoord - 0.5;
            float vignette = 1.0 - dot(center, center) * 0.9;
            
            vec3 shadowTint = vec3(0.03, 0.03, 0.05);
            vec3 highlightTint = vec3(0.80, 0.80, 0.85);
            vec3 tinted = mix(shadowTint, highlightTint, gray);
            vec3 final = tinted * vignette;
            final = pow(final, vec3(1.2));
            
            fragColor = vec4(final, 1.0);
        }
        """

        try:
            vertex_shader = glCreateShader(GL_VERTEX_SHADER)
            glShaderSource(vertex_shader, vertex_source)
            glCompileShader(vertex_shader)

            if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
                error = glGetShaderInfoLog(vertex_shader)
                raise RuntimeError(f"Vertex shader compilation failed: {error}")

            fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(fragment_shader, fragment_source)
            glCompileShader(fragment_shader)

            if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
                error = glGetShaderInfoLog(fragment_shader)
                raise RuntimeError(f"Fragment shader compilation failed: {error}")

            self.shader_program = glCreateProgram()
            glAttachShader(self.shader_program, vertex_shader)
            glAttachShader(self.shader_program, fragment_shader)
            glLinkProgram(self.shader_program)

            if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
                error = glGetProgramInfoLog(self.shader_program)
                raise RuntimeError(f"Shader program linking failed: {error}")

            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)

            self.tex_location = glGetUniformLocation(self.shader_program, "tex")

        except Exception as e:
            logger.error(f"Failed to create shader program: {e}")
            raise

    def _setup_vertex_data(self) -> None:
        try:
            vertices = np.array(
                [
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    -1.0,
                    -1.0,
                    0.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            )

            indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)

            vertex_view = self.vertex_pool.acquire(vertices.nbytes)
            index_view = self.vertex_pool.acquire(indices.nbytes)

            if vertex_view is None or index_view is None:
                raise RuntimeError("Failed to acquire memory from pool")

            vertex_data = np.frombuffer(vertex_view, dtype=np.float32)
            index_data = np.frombuffer(index_view, dtype=np.uint32)
            np.copyto(vertex_data, vertices)
            np.copyto(index_data, indices)

            self.vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(
                GL_ARRAY_BUFFER, vertices.nbytes, vertex_data.tobytes(), GL_STATIC_DRAW
            )

            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER,
                indices.nbytes,
                index_data.tobytes(),
                GL_STATIC_DRAW,
            )

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))

            self.vertex_pool.release(vertices.nbytes, vertex_view)
            self.vertex_pool.release(indices.nbytes, index_view)

        except Exception as e:
            logger.error(f"Failed to setup vertex data: {e}")
            raise

    def update_texture(self, frame_data: np.ndarray) -> None:
        try:
            texture_view = self.vertex_pool.acquire(frame_data.nbytes)
            if texture_view is None:
                raise RuntimeError("Failed to acquire texture memory")

            print("[GL] Copying frame data to GPU")
            texture_data = np.frombuffer(texture_view, dtype=frame_data.dtype).reshape(
                frame_data.shape
            )
            np.copyto(texture_data, frame_data)

            print("[GL] Uploading texture")
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB32F,
                self.width,
                self.height,
                0,
                GL_RGB,
                GL_FLOAT,
                texture_data.tobytes(),
            )
            print("[GL] Texture update complete")

            self.vertex_pool.release(frame_data.nbytes, texture_view)

        except Exception as e:
            logger.error(f"Failed to update texture: {e}")

    def render(self) -> None:
        try:
            print("[GL] Starting frame render")
            glClear(GL_COLOR_BUFFER_BIT)
            glClearColor(0.0, 0.0, 0.0, 1.0)

            glUseProgram(self.shader_program)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glUniform1i(self.tex_location, 0)

            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

            glBindVertexArray(0)
            glUseProgram(0)

            error = glGetError()
            if error != GL_NO_ERROR:
                logger.error(f"OpenGL error: {error}")

        except Exception as e:
            logger.error(f"Failed to render frame: {e}")

    def cleanup(self) -> None:
        try:
            if self.shader_program:
                glDeleteProgram(self.shader_program)
            if self.vao:
                glDeleteVertexArrays(1, [self.vao])
            if self.texture_id:
                glDeleteTextures([self.texture_id])

            self.vertex_pool.cleanup()
            self.texture_manager.cleanup()

        except Exception as e:
            logger.error(f"Failed to clean up resources: {e}")

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
