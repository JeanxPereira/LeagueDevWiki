import sys
import os
import struct
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from PyQt6.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt6.QtCore import QTimer
from PIL import Image


class TroyLoader:
    def __init__(self, filename):
        self.filename = filename
        self.e = []
        self.p = []

        with open(filename, 'rb') as f:
            while True:
                section_size = struct.unpack("<i", f.read(4))[0]
                if section_size == -1:
                    break
                section_type = f.read(4)
                if section_type == b'TXPT':
                    self.load_txpt(f)
                elif section_type == b'ETPT':
                    self.load_etpt(f)
                elif section_type == b'PTPT':
                    self.load_ptpt(f)

    def load_txpt(self, f):
        pass

    def load_etpt(self, f):
        e = {}
        e['name'] = self.read_string(f)
        e['rgba'] = self.read_ubyte4(f)
        e['rate'] = self.read_float(f)
        e['life'] = self.read_float(f)
        e['timeoffset'] = self.read_float(f)
        e['disabled'] = self.read_bool(f)
        e['alpharef'] = self.read_int(f)
        e['rgbaAP1'] = self.read_float2(f)
        e['rgbaAP2'] = self.read_float2(f)
        e['rotation1-axis'] = self.read_float3(f)
        e['rotation2-axis'] = self.read_float3(f)
        e['p-bindtoemitter'] = self.read_float2(f)
        self.e.append(e)

    def load_ptpt(self, f):
        data = f.read(8)
        num_particles, unknown = struct.unpack('<II', data)
        self.log(f"Loading PTPT with {num_particles} particles")
        particles = []
        for i in range(num_particles):
            p = {}
            p['x'] = self.read_float3(f)
            p['scale'] = self.read_float3(f)
            p['rgba'] = self.read_rgba(f)
            p['life'] = self.read_float(f)
            p['linger'] = self.read_float(f)
            p['rotation1'] = self.read_float3(f)
            p['rotation2'] = self.read_float3(f)
            p['unknown1'] = self.read_float(f)
            p['unknown2'] = self.read_float(f)
            p['backfaceon'] = self.read_bool(f)
            p['bindtoemitter'] = self.read_float2(f)
            p['type'] = self.read_int(f)
            p['fresnel'] = self.read_float(f)
            p['fresnel_color'] = self.read_rgba(f)
            p['scaleP1'] = self.read_float2(f)
            p['scaleP2'] = self.read_float2(f)
            p['lifeP1'] = self.read_float2(f)
            p['lifeP2'] = self.read_float2(f)
            p['quadrot'] = self.read_float3(f)
            p['quadrotYP1'] = self.read_float2(f)
            p['quadrotYP2'] = self.read_float2(f)
            p['unknown3'] = self.read_int(f)
            p['unknown4'] = self.read_int(f)
            p['unknown5'] = self.read_int(f)
            p['xrgba'] = self.read_rgba(f)
            p['xrgba1'] = self.read_xrgba(f)
            p['xrgba2'] = self.read_xrgba(f)
            p['xrgba3'] = self.read_xrgba(f)
            p['xrgba4'] = self.read_xrgba(f)
            p['xrgba5'] = self.read_xrgba(f)
            p['xrgba6'] = self.read_xrgba(f)
            p['reflection_fresnel_color'] = self.read_rgba(f)
            p['reflection_opacity_glancing'] = self.read_float(f)
            p['unknown6'] = self.read_bool(f)
            p['rotvel'] = self.read_float3(f)
            p['rotvelYP1'] = self.read_float2(f)
            p['rotvelYP2'] = self.read_float2(f)
            p['mesh'] = self.read_string(f)
            p['meshtex'] = self.read_string(f)
            p['unknown7'] = self.read_int(f)
            p['unknown8'] = self.read_int(f)
            p['unknown9'] = self.read_int(f)
            particles.append(p)
        return particles

class TroyViewer(QMainWindow):
    def init(self):
        super().init()
        self.setWindowTitle("Troy Viewer")
        self.setGeometry(100, 100, 1280, 720)
        self.troyloader = None
        self.current_emitter = None
        self.current_particle = None
        self.current_frame = 0
        self.glwidget = GLWidget(self)
        self.setCentralWidget(self.glwidget)

        # Create menus
        self.file_menu = self.menuBar().addMenu("File")
        self.edit_menu = self.menuBar().addMenu("Edit")
        self.help_menu = self.menuBar().addMenu("Help")

        # Create actions
        self.open_action = self.file_menu.addAction("Open")
        self.exit_action = self.file_menu.addAction("Exit")

        # Connect actions to functions
        self.open_action.triggered.connect(self.open_file)
        self.exit_action.triggered.connect(self.close)

        # Create toolbar
        self.toolbar = self.addToolBar("Tools")
        self.play_action = self.toolbar.addAction("Play")
        self.pause_action = self.toolbar.addAction("Pause")
        self.stop_action = self.toolbar.addAction("Stop")
        self.prev_frame_action = self.toolbar.addAction("Prev Frame")
        self.next_frame_action = self.toolbar.addAction("Next Frame")

        # Connect toolbar actions to functions
        self.play_action.triggered.connect(self.play)
        self.pause_action.triggered.connect(self.pause)
        self.stop_action.triggered.connect(self.stop)
        self.prev_frame_action.triggered.connect(self.prev_frame)
        self.next_frame_action.triggered.connect(self.next_frame)

        # Create status bar
        self.statusBar().showMessage("Ready")

        # Create timer for animation
        self.timer = QTimer()
        self.timer.setInterval(1000 // 60)
        self.timer.timeout.connect(self.update_animation)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Troy File", "", "Troy Files (*.troy)")
        if filename:
            self.troyloader = TroyLoader(filename)
            self.current_emitter = self.troyloader.e[0]
            self.current_particle = self.troyloader.p[0]
            self.glwidget.load_mesh(self.current_particle['mesh'], self.current_particle['meshtex'])
            self.timer.start()

    def play(self):
        self.timer.start()

    def pause(self):
        self.timer.stop()

    def stop(self):
        self.timer.stop()
        self.current_frame = 0
        self.glwidget.set_frame(self.current_frame)

    def prev_frame(self):
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = 0
        self.glwidget.set_frame(self.current_frame)

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame > self.current_particle['num_particles']:
            self.current_frame = self.current_particle['num_particles']
        self.glwidget.set_frame(self.current_frame)

    def update_animation(self):
        self.current_frame += 1
        if self.current_frame >= self.current_particle['num_particles']:
            self.current_frame = 0
        self.glwidget.set_frame(self.current_frame)

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.program = None
        self.texture = None
        self.vao = None
        self.vbo = None

        self.proj_mat = np.eye(4, dtype=np.float32)
        self.view_mat = np.eye(4, dtype=np.float32)
        self.model_mat = np.eye(4, dtype=np.float32)

        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def initializeGL(self):
        self.program = self.create_program()
        self.texture = self.load_texture('particle.png')
        self.vao, self.vbo = self.create_buffers()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.proj_mat = self.create_proj_matrix(45.0, width / height, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glUniformMatrix4fv(glGetUniformLocation(self.program, 'proj_mat'), 1, GL_FALSE, self.proj_mat)
        glUniformMatrix4fv(glGetUniformLocation(self.program, 'view_mat'), 1, GL_FALSE, self.view_mat)
        glUniformMatrix4fv(glGetUniformLocation(self.program, 'model_mat'), 1, GL_FALSE, self.model_mat)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, len(self.particles))
        glBindVertexArray(0)

    def set_particles(self, particles):
        self.particles = particles
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, len(particles) * 3 * 4, particles, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def create_program(self):
        vertex_shader = shaders.compileShader(
            """
            #version 330
            layout(location = 0) in vec3 in_pos;
            layout(location = 1) in vec2 in_uv;
            layout(location = 2) in vec4 in_rgba;
            out vec2 uv;
            out vec4 rgba;
            uniform mat4 proj_mat;
            uniform mat4 view_mat;
            uniform mat4 model_mat;
            void main()
            {
                vec4 pos = vec4(in_pos, 1.0);
                pos = model_mat * pos;
                pos = view_mat * pos;
                pos = proj_mat * pos;
                gl_Position = pos;
                uv = in_uv;
                rgba = in_rgba;
            }
            """,
            GL_VERTEX_SHADER,
        )

        fragment_shader = shaders.compileShader(
            """
            #version 330
            in vec2 uv;
            in vec4 rgba;
            out vec4 frag_color;
            uniform sampler2D texture_sampler;
            void main()
            {
                frag_color = texture(texture_sampler, uv) * rgba;
            }
            """,
            GL_FRAGMENT_SHADER,
        )

        program = shaders.compileProgram(vertex_shader, fragment_shader)
        return program

    def load_texture(self, filename):
        img = Image.open(filename)
        img_data = np.array(list(img.getdata()), np.uint8)
        width, height = img.size
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture, width, height

    def initializeGL(self):
        self.load_shader()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.texture_id is not None and self.particle_buffer is not None:
            glUseProgram(self.shader_program)
            glBindBuffer(GL_ARRAY_BUFFER, self.particle_buffer)
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glEnableVertexAttribArray(2)
            glEnableVertexAttribArray(3)
            glEnableVertexAttribArray(4)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(12))
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 64, ctypes.c_void_p(24))
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(28))
            glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(36))
            glUniformMatrix4fv(self.view_matrix_location, 1, GL_FALSE, self.view_matrix)
            glUniformMatrix4fv(self.projection_matrix_location, 1, GL_FALSE, self.projection_matrix)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glDrawArrays(GL_POINTS, 0, len(self.particles))
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(2)
            glDisableVertexAttribArray(3)
            glDisableVertexAttribArray(4)

    def load_troy(self, filename):
        troy = TroyLoader(filename)
        self.emitters = troy.e
        self.particles = troy.p
        for particle in self.particles:
            self.load_texture(particle['meshtex'])

    def initializeGL(self):
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        self.load_shaders()
        self.load_buffers()

    def load_shaders(self):
        vertex_shader_source = """
        #version 330
        in vec3 vertex_position;
        in vec4 vertex_color;
        in vec2 vertex_texcoord;

        out vec4 fragment_color;
        out vec2 fragment_texcoord;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            gl_Position = projection * view * model * vec4(vertex_position, 1.0);
            fragment_color = vertex_color;
            fragment_texcoord = vertex_texcoord;
        }
        """

        fragment_shader_source = """
        #version 330
        in vec4 fragment_color;
        in vec2 fragment_texcoord;

        out vec4 color;

        uniform sampler2D texture_sampler;

        void main() {
            color = texture(texture_sampler, fragment_texcoord) * fragment_color;
        }
        """

        self.vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        self.fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

        self.shader_program = shaders.compileProgram(self.vertex_shader, self.fragment_shader)

        self.projection_matrix_uniform = glGetUniformLocation(self.shader_program, "projection")
        self.view_matrix_uniform = glGetUniformLocation(self.shader_program, "view")
        self.model_matrix_uniform = glGetUniformLocation(self.shader_program, "model")

        self.vertex_position_attrib = glGetAttribLocation(self.shader_program, "vertex_position")
        self.vertex_color_attrib = glGetAttribLocation(self.shader_program, "vertex_color")
        self.vertex_texcoord_attrib = glGetAttribLocation(self.shader_program, "vertex_texcoord")

    def load_buffers(self):
        e = np.zeros((len(self.e), 8), dtype=np.float32)
        for i, emitter in enumerate(self.e):
            e[i] = [emitter['rate'], emitter['life'], emitter['timeoffset'], 1 if emitter['disabled'] else 0, 
                    emitter['rgba'][0]/255, emitter['rgba'][1]/255, emitter['rgba'][2]/255, emitter['rgba'][3]/255]
        p = np.zeros((len(self.p), 63), dtype=np.float32)
        for i, particle in enumerate(self.p):
            p[i] = [particle['x'][0], particle['x'][1], particle['x'][2], particle['scale'][0], particle['scale'][1], particle['scale'][2],
                    particle['rgba'][0]/255, particle['rgba'][1]/255, particle['rgba'][2]/255, particle['rgba'][3]/255, 
                    particle['life'], particle['linger'], particle['rotation1'][0], particle['rotation1'][1], particle['rotation1'][2], 
                    particle['rotation2'][0], particle['rotation2'][1], particle['rotation2'][2], particle['unknown1'], particle['unknown2'],
                    1 if particle['backfaceon'] else 0, particle['bindtoemitter'][0], particle['bindtoemitter'][1], particle['type'], 
                    particle['fresnel'], particle['fresnel_color'][0]/255, particle['fresnel_color'][1]/255, particle['fresnel_color'][2]/255, 
                    particle['fresnel_color'][3]/255, particle['scaleP1'][0], particle['scaleP1'][1], particle['scaleP2'][0], 
                    particle['scaleP2'][1], particle['lifeP1'][0], particle['lifeP1'][1], particle['lifeP2'][0], particle['lifeP2'][1], 
                    particle['quadrot'][0], particle['quadrot'][1], particle['quadrot'][2], particle['quadrotYP1'][0], 
                    particle['quadrotYP1'][1], particle['quadrotYP2'][0], particle['quadrotYP2'][1], particle['unknown3'], 
                    particle['unknown4'], particle['unknown5'], particle['xrgba'][0], particle['xrgba'][1], particle['xrgba'][2], 
                    particle['xrgba'][3], particle['xrgba1'][0], particle['xrgba1'][1], particle['xrgba1'][2], particle['xrgba1'][3], 
                    particle['xrgba1'][4], particle['xrgba2'][0], particle['xrgba2'][1], particle['xrgba2'][2], particle['xrgba2'][3], 
                    particle['xrgba2'][4], particle['xrgba3'][0], particle['xrgba3'][1], particle['xrgba3'][2], particle['xrgba3'][3], 
                    particle['xrgba3'][4], particle['xrgba4'][0], particle['xrgba4'][1], particle['xrgba4'][2], particle['xrgba4'][3], 
                    particle['xrgba4'][4], particle['xrgba5'][0], particle['xrgba5'][1], particle['xrgba5'][2], particle['xrgba5'][3],
                    particle['xrgba5'][4], particle['xrgba6'][0], particle['xrgba6'][1], particle['mesh'], particle['meshtex']]
            
            # Build the vertices and indices arrays for OpenGL buffers
            vertices = []
            indices = []
            v_idx = 0
            for i, p in enumerate(self.tloader.p):
                mesh = self.tloader.meshes[p['mesh']]
                mesh_verts = mesh['vertices']
                mesh_inds = mesh['indices']
                for j in range(mesh['num_verts']):
                    v_idx = len(vertices)
                    vertex = mesh_verts[j]
                    scale = p['scale']
                    vertices.append([
                        p['x'][0] + vertex[0] * scale[0],
                        p['x'][1] + vertex[1] * scale[1],
                        p['x'][2] + vertex[2] * scale[2],
                        p['rgba'][0], p['rgba'][1], p['rgba'][2], p['rgba'][3],
                        p['quadrot'][0], p['quadrot'][1], p['quadrot'][2],
                        p['quadrotYP1'][0], p['quadrotYP2'][0]
                    ])
                    if j < mesh['num_verts'] - 2:
                        indices.append([v_idx, v_idx + j + 1, v_idx + j + 2])
                if self.tloader.textures.get(p['meshtex']):
                    texture = self.tloader.textures[p['meshtex']]
                    tex_id = texture['id']
                    tex_w = texture['width']
                    tex_h = texture['height']
                    for j in range(mesh['num_verts']):
                        vertices[v_idx + j].append(mesh_verts[j][3] * tex_w)
                        vertices[v_idx + j].append(mesh_verts[j][4] * tex_h)
                        vertices[v_idx + j].append(texture['tx1'])
                        vertices[v_idx + j].append(texture['ty1'])
                        vertices[v_idx + j].append(texture['tx2'])
                        vertices[v_idx + j].append(texture['ty2'])
                else:
                    for j in range(mesh['num_verts']):
                        vertices[v_idx + j].append(0)
                        vertices[v_idx + j].append(0)
                        vertices[v_idx + j].append(0)
                        vertices[v_idx + j].append(0)
                        vertices[v_idx + j].append(0)
                        vertices[v_idx + j].append(0)
            
            # Convert the vertices and indices arrays to NumPy arrays and create the OpenGL buffers
            vdata = np.array(vertices, dtype=np.float32)
            idata = np.array(indices, dtype=np.uint32)
            vbo, ibo = glGenBuffers(2)
            self.buffers = {'vbo': vbo, 'ibo': ibo}
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vdata, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idata, GL_STATIC_DRAW)
            self.buffers['num_indices'] = len(idata)


