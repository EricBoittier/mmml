import bpy
import numpy as np
import mathutils
import matplotlib.pyplot as plt

default_color_dict = {
    "Cl": [102, 227, 115],
    "C": [61, 61, 64],
    "O": [240, 10, 10],
    "N": [10, 10, 240],
    "F": [0, 232, 0],
    "H": [232, 206, 202],
    "K": [128, 50, 100],
    "X": [200, 200, 200],
}

default_color_dict = [None, [232, 206, 202], None, None, None, None, 
[61, 61, 64] , [10, 10, 240], [240, 10, 10], [0, 232, 0]]


# --- Colormap setup using Matplotlib ---
colormap = plt.cm.seismic  # You can change this to other colormaps like 'plasma', 'inferno', etc.
num_colors = 10  # Define how many colors you'd like to sample

# Generate color samples from the colormap
colors = [colormap(i / (num_colors - 1)) for i in range(num_colors)]  # [0, 1] normalized

# Convert RGBA to RGB (Matplotlib includes alpha channel)
colors_rgb = [(r, g, b) for r, g, b, a in colors]


# Create point data



def ESPBLENDER(loaded):
    # --- Clear scene ---
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    points = loaded["esp_grid"]
    values = loaded["esp"]
    vmin, vmax = min(values), max(values)
    scalars = [(v - vmin) / (vmax - vmin) for v in values]

    # Create mesh with scalar attribute
    mesh = bpy.data.meshes.new("PointMesh")
    obj = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(points, [], [])
    mesh.update()

    # Add scalar attribute to mesh
    attr = mesh.attributes.new(name="scalar", type='FLOAT', domain='POINT')
    for i, val in enumerate(scalars):
        attr.data[i].value = val

    # --- Geometry Nodes setup (Separate from Shader Nodes) ---
    mod = obj.modifiers.new("GeometryNodes", 'NODES')
    geo_nodes = bpy.data.node_groups.new("GeoGroup", 'GeometryNodeTree')
    mod.node_group = geo_nodes

    geo_nodes.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    geo_nodes.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = geo_nodes.nodes
    links = geo_nodes.links
    nodes.clear()

    # Nodes for geometry creation
    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-800, 0)

    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (800, 0)

    # Use instance-on-points for visualizing the points
    instancer = nodes.new("GeometryNodeInstanceOnPoints")
    instancer.location = (-400, 0)

    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.location = (-600, -200)
    sphere.inputs["Radius"].default_value = 0.05  # Increase this to make the points overlap
    sphere.inputs["Subdivisions"].default_value = 4

    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (-200, 0)


    # Connect the geometry (points to sphere instances)
    links.new(input_node.outputs["Geometry"], instancer.inputs["Points"])
    links.new(sphere.outputs["Mesh"], instancer.inputs["Instance"])
    links.new(instancer.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], output_node.inputs["Geometry"])

    # --- Material Creation and Assignment ---
    # Create material for the points
    mat = bpy.data.materials.new("ColorByScalar")
    mat.use_nodes = True
    mat_nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links

    for node in mat_nodes:
        if node.name != "Principled BSDF" and node.name != "Material Output":
            mat_nodes.remove(node)

    # Shader: Attribute → ColorRamp → BSDF
    attr_node = mat_nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "scalar"
    attr_node.location = (-400, 0)

    ramp_node = mat_nodes.new("ShaderNodeValToRGB")
    ramp_node.location = (-200, 0)

    # Map colors from colormap to the ColorRamp
    for i, (r, g, b) in enumerate(colors_rgb):
        ramp_node.color_ramp.elements.new(i / (num_colors - 1))  # Add new color stop
        ramp_node.color_ramp.elements[i].color = (r, g, b, 1)  # Set RGB

    bsdf = mat_nodes["Principled BSDF"]
    mat_links.new(attr_node.outputs["Fac"], ramp_node.inputs["Fac"])
    mat_links.new(ramp_node.outputs["Color"], bsdf.inputs["Base Color"])

    # --- Geometry Nodes material assignment node (Geometry Node Tree) ---
    geo_nodes_material = nodes.new("GeometryNodeSetMaterial")
    geo_nodes_material.location = (200, 0)
    #links.new(geo_nodes_material.outputs["Geometry"], output_node.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], geo_nodes_material.inputs["Geometry"])

    # --- Geometry Nodes material assignment node (Geometry Node Tree) ---
    geo_nodes_material = nodes.new("GeometryNodeSetMaterial")
    geo_nodes_material.location = (200, 0)
    links.new(geo_nodes_material.outputs["Geometry"], output_node.inputs["Geometry"])

    # Assign the material in the geometry node tree
    geo_nodes_material.inputs["Material"].default_value = mat

    # --- Camera setup (Orthographic camera) ---

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'ORTHO'

    # Adjust orthographic scale
    camera.data.ortho_scale = 1.0  
    bpy.context.view_layer.objects.active = camera 
    # Set the camera as the scene's active camera
    bpy.context.scene.camera = camera



    links.new(realize.outputs["Geometry"], geo_nodes_material.inputs["Geometry"])

    # --- Light setup ---
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    light.location = (2, -2, 3)
    bpy.context.collection.objects.link(light)

    # --- Finalizing the render settings ---
    # Set the render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # Set background to transparent
    bpy.context.scene.render.film_transparent = True

    atom_positions = loaded["R"]  # Positions in [x, y, z] format
    atom_types = loaded["Z"]     # Atomic numbers or atom types
    print(atom_types)
    print(atom_positions)
    # Load ASE colors for elements (atom types)
    # ASE has a default coloring system that can be used to get element colors
    # This is done by creating an Atoms object and using its `get_chemical_symbols` method
    #elements = np.ones(8, dtype=int) # * atom_types
    #from ase import Atoms
    #atoms = Atoms(elements, atom_positions)
    from ase.data.colors import jmol_colors as jmol
    from ase.data import covalent_radii as covalent_radii
    ans = np.array(atom_types, dtype=int).flatten()
    element_colors = jmol[ans] #[default_color_dict[int(_[0])] for _ in atom_types]# Get colors for each element
    radii = covalent_radii[ans]
    # --- Clear any existing objects ---
    #bpy.ops.object.select_all(action='SELECT')
    #bpy.ops.object.delete(use_global=False)


    # --- Create spheres for each atom ---
    for i, pos in enumerate(atom_positions):
        # Create a sphere mesh for each atom
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=pos)  # Adjust radius for size
        sphere = bpy.context.object
        sphere.name = f"Atom_{i}"
        e = atom_types[i]
        mat = bpy.data.materials.get(f"AtomMaterial_{e}")
        if mat is None:
            # Set the color for the sphere based on the atom's element color
            mat = bpy.data.materials.new(f"AtomMaterial_{e}")
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        
        # Use the RGB color from ASE's element color data
        color = element_colors[i] # Color is a tuple (R, G, B)
        bsdf.inputs["Base Color"].default_value = (*color, 0.8)  # Add alpha channel as 1 (opaque)
            
        # Assign the material to the object
        if sphere.data.materials:
            sphere.data.materials[0] = mat  # Replace the first material
        else:
            sphere.data.materials.append(mat)  # Add the material if there are no materials



    # --- Finalizing the camera view ---
    # Select the mesh and fit the camera to view
    #obj.select_set(True)
    #bpy.ops.object.select_all(action='SELECT')
    #bpy.context.view_layer.objects.active = obj
    #bpy.ops.view3d.camera_to_view_selected()


    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = camera 
    bpy.context.scene.camera = camera
    bpy.ops.view3d.camera_to_view_selected()
    camera.data.ortho_scale = camera.data.ortho_scale * 2.5


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    loaded = np.load(path)
    ESPBLENDER(loaded)