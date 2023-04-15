import argparse
import sys
import os
import json
import math

import numpy as np
import pyglet

class TroyPreviewer:

    def __init__(self, troy_file, model_dir=None, img_dir=None):
        self.troy_file = troy_file
        self.model_dir = model_dir
        self.img_dir = img_dir

        self.load_troy()
        self.setup_window()

    def load_troy(self):
        # TODO: Implement troy file loading
        pass

    def setup_window(self):
        # TODO: Implement window setup
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preview League of Legends .troy files.')
    parser.add_argument('troy_file', type=str, help='input .troy file name')
    parser.add_argument('--model_dir', type=str, help='directory containing 3D models used in particles')
    parser.add_argument('--img_dir', type=str, help='directory containing images used in particles')
    args = parser.parse_args()

    previewer = TroyPreviewer(args.troy_file, args.model_dir, args.img_dir)
    pyglet.app.run()

def read_troy(troy_file_path):
    with open(troy_file_path, 'rb') as f:
        data = f.read()

    magic_number = struct.unpack('i', data[:4])[0]
    if magic_number != 0x13579:
        raise ValueError('Invalid magic number')

    version = struct.unpack('i', data[4:8])[0]
    if version != 2:
        raise ValueError('Invalid version number')

    header_length = struct.unpack('i', data[8:12])[0]
    header_data = data[12:12+header_length]

    header = json.loads(header_data)

    return header, data[12+header_length:]

def preview_troy(troy_file_path, image_path, model_path):
    header, particle_data = read_troy(troy_file_path)

    image = pyglet.image.load(image_path)
    model = pyglet.model.load(model_path)

    window = pyglet.window.Window(header['ParticleWidth'], header['ParticleHeight'], caption='Troy Preview')
    glClearColor(0, 0, 0, 1)

    particle_batch = pyglet.graphics.Batch()

    for particle in header['Particles']:
        sprite = pyglet.sprite.Sprite(image, batch=particle_batch)

        x, y, z = particle['Position']
        sprite.position = (x, y)
        sprite.scale = particle['Scale']
        sprite.opacity = particle['Alpha']

        rotation_x, rotation_y, rotation_z = particle['Rotation']
        sprite.rotation = -rotation_z  # convert from clockwise to counterclockwise rotation

        particle_model = pyglet.model.add_simple_model(1.0, 1.0, 1.0, model, batch=particle_batch)

        particle_model.position = (x, y, z)
        particle_model.scale = particle['Scale']
        particle_model.opacity = particle['Alpha']
        particle_model.rotation_x = rotation_x
        particle_model.rotation_y = rotation_y
        particle_model.rotation_z = rotation_z

    @window.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT)
        particle_batch.draw()

    pyglet.app.run()

window = pyglet.window.Window(resizable=True)

batch = pyglet.graphics.Batch()
particle_sprites = []

# Define the vertex list for a particle
def create_particle_vertex_list(position, size):
    x, y, z = position
    size_x, size_y = size
    return batch.add(4, GL_QUADS, None, ('v3f', [
        x - size_x / 2, y - size_y / 2, z,
        x + size_x / 2, y - size_y / 2, z,
        x + size_x / 2, y + size_y / 2, z,
        x - size_x / 2, y + size_y / 2, z]), ('c4B', [255, 255, 255, 255] * 4))

# Load the textures for the particles
def load_particle_textures(textures):
    loaded_textures = {}
    for name, path in textures.items():
        loaded_textures[name] = pyglet.image.load(path).texture
    return loaded_textures

# Load the 3D models for the particles
def load_particle_models(models):
    loaded_models = {}
    for name, path in models.items():
        loaded_models[name] = pyglet.resource.model(path)
    return loaded_models

# Define a class to hold the information about a single particle
class Particle:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes
        self.texture = None
        self.model = None
        self.sprite = None
        self.update_sprite()

    # Update the sprite for this particle based on its current attributes
    def update_sprite(self):
        if self.sprite is not None:
            self.sprite.delete()

        # Get the position and size of the particle
        position = (self.attributes["Position"]["X"], self.attributes["Position"]["Y"], self.attributes["Position"]["Z"])
        size = (self.attributes["Size"]["X"], self.attributes["Size"]["Y"])

        # Create a new sprite for the particle
        if self.attributes["RenderType"] == "Billboard":
            self.texture = self.texture or loaded_textures[self.attributes["TextureName"]]
            self.sprite = pyglet.sprite.Sprite(self.texture, *position, batch=batch)
            self.sprite.scale_x = size[0] / self.texture.width
            self.sprite.scale_y = size[1] / self.texture.height
        elif self.attributes["RenderType"] == "Model":
            self.model = self.model or loaded_models[self.attributes["ModelName"]]
            self.sprite = pyglet.sprite.Sprite(self.model, *position, batch=batch)
            self.sprite.scale = self.attributes["ModelScale"]
        elif self.attributes["RenderType"] == "Beam":
            pass # TODO: Implement beam rendering
        else:
            raise Exception("Unknown RenderType: {}".format(self.attributes["RenderType"]))

    # Set a new attribute value for this particle
    def set_attribute(self, name, value):
        self.attributes[name] = value
        self.update_sprite()


    # Set up window and OpenGL context
    win = pyglet.window.Window(800, 600, caption='Troy Preview')
    glClearColor(0, 0, 0, 1)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Load particle textures
    particle_textures = {}
    for texture_name in textures:
        texture_path = os.path.join(args.texture_dir, texture_name + ".png")
        particle_textures[texture_name] = image.load(texture_path).texture

    # Set up particle batch
    particle_batch = pyglet.graphics.Batch()

    # Load particle data
    with open(args.troy_file, "rb") as f:
        troy_data = f.read()

    particle_data = troybin.read(io.BytesIO(troy_data))

    # Create particle sprites
    particles = []
    for particle in particle_data:
        texture_name = particle["Texture"]
        texture = particle_textures[texture_name]
        x, y, z = particle["Position"]
        size = particle["Size"]
        angle = particle["Angle"]
        particles.append(pyglet.sprite.Sprite(texture, x=x, y=y, batch=particle_batch))

    # Set up camera
    camera_x = 0
    camera_y = 0
    camera_z = 0
    camera_rot_x = 0
    camera_rot_y = 0

    # Set up keyboard/mouse input
    keys = pyglet.window.key.KeyStateHandler()
    win.push_handlers(keys)
    mouse = pyglet.window.mouse
    win.set_mouse_visible(False)

    # Set up update function
    def update(dt):
        global camera_x, camera_y, camera_z, camera_rot_x, camera_rot_y
        if keys[pyglet.window.key.W]:
            camera_x += 100 * dt * math.sin(math.radians(camera_rot_y))
            camera_z += 100 * dt * math.cos(math.radians(camera_rot_y))
        if keys[pyglet.window.key.S]:
            camera_x -= 100 * dt * math.sin(math.radians(camera_rot_y))
            camera_z -= 100 * dt * math.cos(math.radians(camera_rot_y))
        if keys[pyglet.window.key.A]:
            camera_x += 100 * dt * math.sin(math.radians(camera_rot_y - 90))
            camera_z += 100 * dt * math.cos(math.radians(camera_rot_y - 90))
        if keys[pyglet.window.key.D]:
            camera_x += 100 * dt * math.sin(math.radians(camera_rot_y + 90))
            camera_z += 100 * dt * math.cos(math.radians(camera_rot_y + 90))
        if keys[pyglet.window.key.SPACE]:
            camera_y += 100 * dt
        if keys[pyglet.window.key.LSHIFT]:
            camera_y -= 100 * dt
        dx, dy = mouse.get_delta()
        camera_rot_x += dy * 0.1
        camera_rot_y -= dx * 0.1
        camera_rot_x = max(min(camera_rot_x, 90), -90)
        glLoadIdentity()
        glTranslatef(-camera_x, -camera_y, -camera_z)
        glRotatef(camera_rot_x, 1, 0, 0)
        glRotatef(camera_rot_y, 0, 1, 0)

    # Set up drawing function
    def draw():
        win.clear()
        particle_batch.draw()

    # Run main loop
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()

    # Load the 3D model of the particle
    particle_model = pyglet.resource.model('particle.obj')

    # Load the image to be used as a texture
    particle_texture = pyglet.resource.image('particle_texture.png')

    # Create the batch for the particles
    particle_batch = pyglet.graphics.Batch()

    # Create a list of particle instances
    particles = []

    # Create a function to add a particle to the batch
    def add_particle(x, y, z):
        particle = pyglet.graphics.Batch()
        particle.add(len(particle_model.vertex_lists), pyglet.gl.GL_TRIANGLES, None,
            ('v3f', particle_model.vertices),
            ('t2f', particle_model.tex_coords)
        )
        particles.append((x, y, z, particle))

    # Call the function to add particles at various positions
    add_particle(0, 0, 0)
    add_particle(1, 0, 0)
    add_particle(0, 1, 0)
    add_particle(0, 0, 1)

    # Create a function to update the positions of the particles
    def update_particles(dt):
        pass  # TODO: Implement this function

    # Create the window
    window = pyglet.window.Window(width=800, height=600, caption='Particle Preview')

    # Set up the camera
    camera_pos = Vector4([0, 0, -5, 1])
    camera_rot = Vector4([0, 0, 0, 1])
    view = Matrix44.look_at(
        camera_pos.xyz,
        [0, 0, 0],
        [0, 1, 0]
    )
    projection = Matrix44.perspective_projection(45.0, window.width / window.height, 0.1, 100.0)

    # Set up the OpenGL context
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Set up the event loop
    @window.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(projection.astype('f4').flatten())
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(view.astype('f4').flatten())

        particle_texture.blit(0, 0)  # TODO: Use the texture for the particles
        for x, y, z, particle in particles:
            glPushMatrix()
            glTranslatef(x, y, z)
            particle.draw()
            glPopMatrix()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            window.close()

    pyglet.clock.schedule_interval(update_particles, 1 / 60.0)
    pyglet.app.run()

# Part 5: Define the particle system class
class ParticleSystem:
    def __init__(self, troy_file):
        self.particle_list = []
        self.frame_duration = 0.03  # default frame duration
        self.frame_count = 0
        self.loop = True
        self.load_troy_file(troy_file)

    def load_troy_file(self, troy_file):
        # Load troy file and create particle objects
        pass

    def update(self, dt):
        # Update particle positions and frames
        pass

    def draw(self):
        # Draw particles
        pass

# Part 6: Load images and 3D models for the particles
particle_images = {}
particle_models = {}

# Load particle images
for image_file in os.listdir("particle_images"):
    image_path = os.path.join("particle_images", image_file)
    particle_images[image_file] = pyglet.image.load(image_path)

# Load particle models
for model_file in os.listdir("particle_models"):
    model_path = os.path.join("particle_models", model_file)
    # Load model and add it to particle_models dictionary
    pass

# Part 7: Create the pyglet window and define event handlers
class ParticleViewerWindow(pyglet.window.Window):
    def __init__(self, particle_system):
        super().__init__(caption="Particle Viewer")
        self.particle_system = particle_system
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

    def on_draw(self):
        self.clear()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.particle_system.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            pyglet.app.exit()

    def on_close(self):
        pyglet.app.exit()

# Part 8: Start the pyglet application
if __name__ == "__main__":
    troy_file = "particle_system.troy"
    particle_system = ParticleSystem(troy_file)
    window = ParticleViewerWindow(particle_system)
    pyglet.clock.schedule_interval(particle_system.update, particle_system.frame_duration)
    pyglet.app.run()

# Create a window
window = pyglet.window.Window(800, 600, resizable=True)

# Create a 3D projection
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45.0, window.width / window.height, 0.1, 100.0)
glMatrixMode(GL_MODELVIEW)

# Set the camera position
glLoadIdentity()
gluLookAt(0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# Enable depth testing
glEnable(GL_DEPTH_TEST)

# Set the clear color to black
glClearColor(0.0, 0.0, 0.0, 0.0)

# Load the particle texture
particle_image = pyglet.resource.image('particle.png')
particle_texture = particle_image.get_texture()
particle_texture.width = particle_image.width
particle_texture.height = particle_image.height
glEnable(GL_TEXTURE_2D)

# Set the blend function
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Create a batch for the particles
particle_batch = pyglet.graphics.Batch()

# Define the particle vertices
particle_vertices = [
    -1.0, -1.0, 0.0, 0.0, 0.0,
    1.0, -1.0, 0.0, 1.0, 0.0,
    1.0, 1.0, 0.0, 1.0, 1.0,
    -1.0, 1.0, 0.0, 0.0, 1.0
]

# Define the particle indices
particle_indices = [
    0, 1, 2,
    0, 2, 3
]

# Create a buffer for the particle vertices and indices
particle_vbo = pyglet.graphics.vertexbuffer.create_buffer(len(particle_vertices) * 4, GL_ARRAY_BUFFER, GL_STATIC_DRAW)
particle_vbo.bind()
particle_vbo.set_data(particle_vertices)

particle_ibo = pyglet.graphics.vertexbuffer.create_buffer(len(particle_indices) * 2, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW)
particle_ibo.bind()
particle_ibo.set_data(particle_indices)

# Set the vertex and texture coordinate pointers
glEnableClientState(GL_VERTEX_ARRAY)
glEnableClientState(GL_TEXTURE_COORD_ARRAY)
glVertexPointer(3, GL_FLOAT, 20, particle_vbo)
glTexCoordPointer(2, GL_FLOAT, 20, particle_vbo + 12)

# Define the particle update function
def update_particles(dt):
    # Update the particle positions and colors here
    pass

# Define the particle draw function
def draw_particles():
    # Set the particle texture
    glBindTexture(GL_TEXTURE_2D, particle_texture.id)

    # Set the particle color
    glColor4f(1.0, 1.0, 1.0, 1.0)

    # Draw the particles
    particle_batch.draw()

# Initialize the scene
scene = bpy.context.scene
scene.render.fps = fps
scene.frame_end = frame_count
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.color_depth = '16'
scene.render.image_settings.compression = 15

# Create a camera object and position it
camera_data = bpy.data.cameras.new(name="Camera")
camera_obj = bpy.data.objects.new(name="Camera", object_data=camera_data)
scene.camera = camera_obj
camera_obj.location = camera_position
camera_obj.rotation_euler = camera_rotation

# Create a light object and position it
light_data = bpy.data.lights.new(name="Light", type='POINT')
light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
light_obj.location = light_position
light_obj.data.energy = light_intensity
scene.collection.objects.link(light_obj)

# Create a particle system and add particles
particle_system = bpy.data.particlesystems.new(name="Particle System")
particle_system.vertex_group_density = "density"
particle_system.vertex_group_size = "size"
emitter_object.modifiers.new(name="Particle System", type='PARTICLE_SYSTEM')
emitter_object.particle_systems[-1] = particle_system
particle_settings = particle_system.settings
particle_settings.type = 'HAIR'
particle_settings.use_advanced_hair = True
particle_settings.count = particle_count
particle_settings.hair_length = hair_length
particle_settings.radius_scale = radius_scale
particle_settings.rendered_steps = render_steps
particle_settings.display_step = display_step
particle_settings.hair_step = hair_step
particle_settings.keyed_loops = keyed_loops
particle_settings.use_close_tip = use_close_tip
particle_settings.use_rotations = use_rotations
particle_settings.rotation_mode = 'NONE'
particle_settings.use_velocity_length = use_velocity_length
particle_settings.use_emit_random = use_emit_random
particle_settings.emit_random = emit_random
particle_settings.hair_step = hair_step
particle_settings.clump_factor = clump_factor
particle_settings.clump_shape = clump_shape
particle_settings.kink_amplitude = kink_amplitude
particle_settings.kink_frequency = kink_frequency
particle_settings.use_kink_shape = use_kink_shape
particle_settings.kink_shape = kink_shape
particle_settings.use_children = use_children
particle_settings.child_type = 'SIMPLE'
particle_settings.child_nbr = child_nbr
particle_settings.child_radius = child_radius
particle_settings.child_length = child_length
particle_settings.child_roundness = child_roundness
particle_settings.use_parent_particles = use_parent_particles
particle_settings.use_hair_bspline = use_hair_bspline

# Create a material and add a texture
material = bpy.data.materials.new(name="Material")
texture = bpy.data.textures.new(name="Texture", type='IMAGE')
texture.image = bpy.data.images.load(image_path)
texture_slot = material.texture_slots.add()
texture_slot.texture = texture
texture_slot.texture_coords = 'UV'
texture_slot.blend_type = 'MULTIPLY'
texture_slot.use_map_color_diffuse = True
texture_slot.diffuse_color_factor = diffuse_color_factor
texture_slot.use_map_alpha = True
texture_slot.alpha_factor = alpha_factor
texture_slot.use_map_density = True
texture_slot.density_factor = density_factor
material.diffuse_color = diffuse_color
material.alpha = alpha
material.specular_intensity = specular_intensity
material.specular_hardness = specular_hardness
material.use_shadeless = use_shadeless

# Assign the material to the emitter object
emitter_object.data.materials.append(material)

class Particle:
    def __init__(self, name, lifespan, emitter, geometry, texture):
        self.name = name
        self.lifespan = lifespan
        self.emitter = emitter
        self.geometry = geometry
        self.texture = texture

    def draw(self):
        # Draw the particle using its geometry and texture
        pass

class ParticleEmitter:
    def __init__(self, name, position, rotation, scale, particles):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.particles = particles

    def update(self, delta_time):
        # Update the position and rotation of the emitter
        # Update the particles of the emitter
        pass

class ParticleSystem:
    def __init__(self, name, emitters):
        self.name = name
        self.emitters = emitters

    def update(self, delta_time):
        # Update all the emitters of the system
        pass

    def draw(self):
        # Draw all the particles of the system
        pass

def load_troy_file(filename):
    # Load the troy file and create a ParticleSystem
    pass

def main():
    # Load the troy file and create a ParticleSystem
    # Create a window and a camera
    # Load the 3D models and textures of the particles
    # Start the main loop
    pass

if __name__ == '__main__':
    main()

def create_particle_geometry(attributes):
    """Create the geometry for the particle based on its attributes"""
    # TODO: Implement the function to create particle geometry
    pass

def load_particle_texture(texture_path):
    """Load the texture image for the particle"""
    # TODO: Implement the function to load particle texture
    pass

def create_particle_mesh(geometry, texture):
    """Create the particle mesh using the given geometry and texture"""
    # TODO: Implement the function to create particle mesh
    pass

def animate_particle(mesh, attributes):
    """Animate the particle mesh based on its attributes"""
    # TODO: Implement the function to animate particle mesh
    pass

# Parse command line arguments
parser = argparse.ArgumentParser(description='Preview a League of Legends particle')
parser.add_argument('particle', type=str, help='path to particle .troy file')
parser.add_argument('--model', type=str, help='path to 3D model file')
parser.add_argument('--texture', type=str, help='path to particle texture image file')
args = parser.parse_args()

# Load particle data from file
with open(args.particle, 'rb') as f:
    particle_data = troybin.read(f)

# Create particle geometry
geometry = create_particle_geometry(particle_data['Attributes'])

# Load particle texture
if args.texture:
    texture = load_particle_texture(args.texture)
else:
    texture = None

# Create particle mesh
mesh = create_particle_mesh(geometry, texture)

# Animate particle mesh
animate_particle(mesh, particle_data['Attributes'])

# Import models
model_folder = "models/"
model_files = {"default": "sphere.obj", "sprite": "sprite.obj"}
models = {}
for key, value in model_files.items():
    model_path = os.path.join(model_folder, value)
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            models[key] = f.read()

# Load textures
texture_folder = "textures/"
texture_files = {"default": "default.png"}
textures = {}
for key, value in texture_files.items():
    texture_path = os.path.join(texture_folder, value)
    if os.path.exists(texture_path):
        textures[key] = imageio.imread(texture_path)

def main(args):
    parser = argparse.ArgumentParser(description='Preview .troy files in 3D')
    parser.add_argument('infile', type=str, help='input .troy file name')
    parser.add_argument('--imgdir', type=str, default='', help='directory containing images referenced by the .troy file')
    args = parser.parse_args()

    if args.imgdir:
        pyglet.resource.path.append(args.imgdir)
        pyglet.resource.reindex()

    window = pyglet.window.Window(800, 600, resizable=True)

    def on_resize(width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, float(width)/height, 0.1, 1000)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    @window.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, -5)
        glRotatef(-90, 1, 0, 0)
        glRotatef(-90, 0, 0, 1)
        glScalef(0.01, 0.01, 0.01)
        batch.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            glRotatef(dx, 0, 1, 0)
            glRotatef(dy, 1, 0, 0)

    ibin = read_troy(args.infile)
    fix_troy(ibin)
    textures = load_textures(ibin)

    batch = pyglet.graphics.Batch()
    for i, emitter in enumerate(ibin["EMITTERS"]):
        group = pyglet.graphics.Group()
        vertices, colors, tex_coords = create_particles(emitter, textures)
        particle_count = len(emitter["POSITIONS"])
        if particle_count:
            batch.add_indexed(particle_count, pyglet.gl.GL_TRIANGLES, group, emitter["INDICES"], ('v3f/static', vertices), ('c3f/static', colors), ('t2f/static', tex_coords))
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)

    pyglet.app.run()

if __name__ == '__main__':
    main(sys.argv[1:])
