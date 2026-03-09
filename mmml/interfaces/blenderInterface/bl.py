import bpy
import sys
import sys
import subprocess

# subprocess.run([sys.executable, '-m', 'pip', 'install', 'ase', '-t',
#    'C:\\Users\\Eric\\AppData\\Roaming\\Blender Foundation\\Blender\\4.4\\scripts\\modules'])
sys.path.append(r"c:\users\eric\appdata\roaming\python\python311\site-packages")

import ase.io
import numpy as np

import bpy
import numpy as np

import itertools

import molecularnodes as mn
import bpy
import numpy as np
from mathutils import Vector


def create_arrow(start, end, name="ForceArrow"):
    """Create a visible 3D arrow between two points."""
    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.05 / 100  # Adds thickness to make it visible
    curve_data.bevel_resolution = 3  # Smooths out the curve

    spline = curve_data.splines.new(type="POLY")
    spline.points.add(1)
    spline.points[0].co = (*start, 1)
    spline.points[1].co = (*end, 1)

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    # Create and assign material
    mat = bpy.data.materials.get(name + "Material")
    if mat is None:
        mat = bpy.data.materials.new(name=name + "Material")
        mat.diffuse_color = (1.0, 0.0, 1.0, 1.0)  # Red color for visibility

    obj.data.materials.append(mat)
    # Create the arrowhead (cone)
    bpy.ops.mesh.primitive_cone_add(radius1=0.1 / 100, depth=0.09 / 100, location=end)
    arrowhead = bpy.context.object
    arrowhead.name = f"{name}_Head"
    # Align arrowhead in the same direction
    arrowhead.rotation_mode = "QUATERNION"
    # Convert direction to a Blender Vector
    direction = np.array(end) - np.array(start)
    dir_vector = Vector(direction)
    arrowhead.rotation_quaternion = dir_vector.to_track_quat("Z", "Y")
    arrowhead.data.materials.append(mat)
    return obj


import bpy


def delete_all_objects():
    """Deletes all objects in the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def add_orthographic_camera():
    """Adds an orthographic camera to the scene."""
    # Create a new camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"

    # Adjust orthographic scale
    camera.data.ortho_scale = 10.0
    bpy.context.view_layer.objects.active = camera
    return camera


def best_fit_rotation_matrix(P, Q):
    """
    Computes the best-fit rotation matrix that aligns two sets of 3D points P and Q.

    :param P: (N, 3) array of source points
    :param Q: (N, 3) array of target points
    :return: (3, 3) rotation matrix R
    """
    # Compute the covariance matrix
    H = P.T @ Q

    # Compute Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = 1, not -1 which would be a reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    return R


import bpy


def create_3d_text(text, location=(0, 0, 0), size=1):
    """Create a 3D text object in the scene."""
    # Create text object
    # Create a new material
    material = bpy.data.materials.new(name="TextMaterial")
    material.use_nodes = True  # Enable nodes for the material
    # Set the base color of the material
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)

    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.data.body = text  # Set the text content
    obj.scale = (size, size, size)  # Set the size of the text

    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = material  # Replace the first material
    else:
        obj.data.materials.append(
            material
        )  # Add the material if there are no materials

    return obj


# Example usage:
delete_all_objects()

# bpy.data.node_groups["MN_NewTrajectory.001"].nodes["Style Ball and Stick"].inputs[7].default_value = 0.1
# bpy.context.space_data.context = 'MATERIAL'
# bpy.data.materials["MN Default"].node_tree.nodes["Glass BSDF"].inputs[2].default_value = 0.1
# bpy.context.space_data.context = 'WORLD'


import molecularnodes.entities.trajectory as mnt


CURRENT_FRAME = 0


def load_traj():
    traj = mnt.load(
        r"C:\Users\Eric\Downloads\ase\test.pdb",
        r"C:\Users\Eric\Downloads\ase\test.xyz",
        style="ball+stick",
    )

    # for CURRENT_FRAME in range(len(traj.universe.trajectory[:3])):
    traj = mnt.load(
        r"C:\Users\Eric\Downloads\ase\test.pdb",
        r"C:\Users\Eric\Downloads\ase\test.xyz",
        name="NewTrajectory",
        style="ball+stick",
    )
    # bpy.data.node_groups["Style Ball and Stick"].nodes].input[4].default_value = 0.1
    # bpy.data.node_groups["Style Ball and Stick"].interface.items_tree["Radius"] = 0.01
    # for k in bpy.data.node_groups["Style Ball and Stick"].nodes.keys():
    #    print(k)
    #    if len(bpy.data.node_groups["Style Ball and Stick"].nodes[k].inputs) > 4:
    #        bpy.data.node_groups["Style Ball and Stick"].nodes[k].inputs[4].default_value = 0.1

    bpy.context.scene.frame_set(CURRENT_FRAME)

    asetraj = ase.io.read(
        r"C:\Users\Eric\Downloads\ase\merged.traj", index=f"{CURRENT_FRAME}"
    )

    atoms = asetraj
    fs = atoms.get_forces()
    e = atoms.get_potential_energy()
    pos = atoms.get_positions()
    posmean = pos.T.mean(axis=1)
    pos = pos - posmean

    R = best_fit_rotation_matrix(pos, traj.universe.trajectory[CURRENT_FRAME].positions)

    fs = fs @ R.T

    for i in range(len(atoms)):
        start = traj.universe.trajectory[CURRENT_FRAME].positions[i]
        end = start + fs[i]
        create_arrow(start / 100, end / 100, name="ForceArrow")

    sc = bpy.context.scene

    # Set output resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    # Set file format and output path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    # Enable transparent background
    bpy.context.scene.render.film_transparent = True

    # Set file format to PNG (supports transparency)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = (
        "RGBA"  # Ensure alpha channel is included
    )

    # Enable GPU if available
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
        "CUDA"  # or 'OPTIX' / 'HIP' for AMD
    )

    m = traj.universe.trajectory[CURRENT_FRAME].positions.T.mean(axis=1)
    m /= 100
    m[0] -= m[0] / 3
    m[1] += m[1] / 5
    m[2] += m[2] / 3.5
    # Example usage: create a 3D text at the origin
    text_obj = create_3d_text(
        f"ID:\t{CURRENT_FRAME}\nE:\t{e:.3e}", location=(m[0], m[1], m[2]), size=0.005
    )

    camera = add_orthographic_camera()
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = camera
    bpy.context.scene.camera = camera
    bpy.ops.view3d.camera_to_view_selected()
    camera.data.ortho_scale = camera.data.ortho_scale * 1.2
    r = r"C:\Users\Eric\Downloads\ase\\"
    r = r + str(CURRENT_FRAME) + "test-data.png"

    sc.render.filepath = r

    bpy.ops.render.render("INVOKE_DEFAULT", write_still=True)

    bpy.data.images["Render Result"].save_render(filepath=sc.render.filepath)


load_traj()
