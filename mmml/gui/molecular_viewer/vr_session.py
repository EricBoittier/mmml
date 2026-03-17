"""
OpenXR VR session for molecular viewer.
Uses pyopenxr's ContextObject with EGL (Wayland) or GLFW/GLX (X11) offscreen context.
Controller: left thumbstick = pan, right thumbstick Y = zoom, thumbstick click = recenter.
View modes: X, Y (left), A (right) = align molecule to X, Y, Z axis.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import xr
from OpenGL.GL import glRotatef, glTranslatef

from xr.utils import rotation_from_quaternionf, view_matrix_from_posef
from xr.utils.gl.context_object import ContextObject

from .debug import debug
from .molecule import Atom, center_and_scale, compute_bonds
from .renderer import (
    clear_screen,
    draw_annotations,
    draw_molecule,
    draw_overlay,
    draw_pointer_ray,
    draw_pbc_cell,
    init_gl,
    setup_projection_vr,
    setup_view_vr,
)


def _get_context_provider():
    """Use EGL on Wayland (xDisplay NULL with GLX), GLFW/GLX on X11."""
    try:
        from xr.utils.gl.egl_util import EGLOffscreenContextProvider
        debug("Using EGL offscreen context (Wayland)")
        return EGLOffscreenContextProvider()
    except Exception as e:
        debug("EGL failed (%s), falling back to GLFW/GLX", e)
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    debug("Using GLFW/GLX offscreen context")
    return GLFWOffscreenContextProvider(gl_version=(4, 1))


def _get_extensions():
    """Extensions for OpenXR instance. Prefer EGL (Wayland) when available."""
    exts = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
    try:
        exts.append(xr.MNDX_EGL_ENABLE_EXTENSION_NAME)
    except AttributeError:
        pass
    return exts


def run_vr_session(
    frames: list[list[Atom]],
    cells: list | None = None,
    metadata: list | None = None,
    atom_scale: float = 0.25,
    scale: float = 1.0,
    molecule_offset_z: float = -2.0,
    playback_fps: float = 10.0,
    pointer: bool = True,
    switch_to_desktop: list | None = None,
) -> None:
    """
    Run the OpenXR VR render loop.
    - frames: list of atom lists (trajectory)
    - atom_scale: atom sphere scale
    - scale: molecule scale
    - molecule_offset_z: push molecule in front of user (meters)
    - playback_fps: trajectory frame advance rate (if multiple frames)
    """
    context_provider = _get_context_provider()
    debug("Creating OpenXR instance and session")

    try:
        instance_create_info = xr.InstanceCreateInfo(
            create_flags=xr.InstanceCreateFlags(),
            application_info=xr.ApplicationInfo(
                application_name="Molecular Viewer",
                application_version=xr.Version(0, 1, 0),
                engine_name="molecular_viewer",
                engine_version=xr.Version(0, 1, 0),
                api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH),
            ),
            enabled_api_layer_names=[],
            enabled_extension_names=_get_extensions(),
        )

        with ContextObject(
            context_provider=context_provider,
            instance_create_info=instance_create_info,
        ) as ctx:
            init_gl()

            # Controller actions: move (pan) and zoom
            hand_left = xr.string_to_path(ctx.instance, "/user/hand/left")
            hand_right = xr.string_to_path(ctx.instance, "/user/hand/right")
            move_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="move",
                    action_type=xr.ActionType.VECTOR2F_INPUT,
                    count_subaction_paths=2,
                    subaction_paths=[hand_left, hand_right],
                    localized_action_name="Move",
                ),
            )
            zoom_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="zoom",
                    action_type=xr.ActionType.VECTOR2F_INPUT,
                    count_subaction_paths=2,
                    subaction_paths=[hand_left, hand_right],
                    localized_action_name="Zoom",
                ),
            )
            recenter_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="recenter",
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    count_subaction_paths=2,
                    subaction_paths=[hand_left, hand_right],
                    localized_action_name="Recenter",
                ),
            )
            view_x_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="view_x",
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    count_subaction_paths=1,
                    subaction_paths=[hand_left],
                    localized_action_name="View X",
                ),
            )
            view_y_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="view_y",
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    count_subaction_paths=1,
                    subaction_paths=[hand_left],
                    localized_action_name="View Y",
                ),
            )
            view_z_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="view_z",
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    count_subaction_paths=1,
                    subaction_paths=[hand_left],
                    localized_action_name="View Z",
                ),
            )
            pointer_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="pointer",
                    action_type=xr.ActionType.POSE_INPUT,
                    count_subaction_paths=2,
                    subaction_paths=[hand_left, hand_right],
                    localized_action_name="Pointer",
                ),
            )
            pointer_space_left = xr.create_action_space(ctx.session, xr.ActionSpaceCreateInfo(pointer_action, hand_left))
            pointer_space_right = xr.create_action_space(ctx.session, xr.ActionSpaceCreateInfo(pointer_action, hand_right))
            toggle_text_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="toggle_text",
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    count_subaction_paths=2,
                    subaction_paths=[hand_left, hand_right],
                    localized_action_name="Toggle Text",
                ),
            )
            switch_mode_action = xr.create_action(
                ctx.default_action_set,
                xr.ActionCreateInfo(
                    action_name="switch_mode",
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    count_subaction_paths=2,
                    subaction_paths=[hand_left, hand_right],
                    localized_action_name="Switch Mode",
                ),
            )
            # Suggest bindings for Oculus Touch (Rift S). Simple controller has no thumbstick.
            for profile, bindings in [
                (
                    "/interaction_profiles/oculus/touch_controller",
                    [
                        (move_action, "/user/hand/left/input/thumbstick"),
                        (zoom_action, "/user/hand/right/input/thumbstick"),
                        (recenter_action, "/user/hand/left/input/thumbstick/click"),
                        (recenter_action, "/user/hand/right/input/thumbstick/click"),
                        (view_x_action, "/user/hand/left/input/x/click"),
                        (view_y_action, "/user/hand/left/input/y/click"),
                        (view_z_action, "/user/hand/right/input/a/click"),
                        (pointer_action, "/user/hand/left/input/aim/pose"),
                        (pointer_action, "/user/hand/right/input/aim/pose"),
                        (toggle_text_action, "/user/hand/right/input/b/click"),
                        (switch_mode_action, "/user/hand/left/input/menu/click"),
                        (switch_mode_action, "/user/hand/left/input/grip/click"),
                    ],
                ),
                (
                    "/interaction_profiles/valve/index_controller",
                    [
                        (move_action, "/user/hand/left/input/thumbstick"),
                        (zoom_action, "/user/hand/right/input/thumbstick"),
                        (recenter_action, "/user/hand/left/input/thumbstick/click"),
                        (recenter_action, "/user/hand/right/input/thumbstick/click"),
                        (view_x_action, "/user/hand/left/input/a/click"),
                        (view_y_action, "/user/hand/left/input/b/click"),
                        (view_z_action, "/user/hand/right/input/a/click"),
                        (pointer_action, "/user/hand/left/input/aim/pose"),
                        (pointer_action, "/user/hand/right/input/aim/pose"),
                        (toggle_text_action, "/user/hand/right/input/b/click"),
                        (switch_mode_action, "/user/hand/left/input/system/click"),
                        (switch_mode_action, "/user/hand/left/input/grip/click"),
                    ],
                ),
            ]:
                try:
                    xr.suggest_interaction_profile_bindings(
                        ctx.instance,
                        xr.InteractionProfileSuggestedBinding(
                            interaction_profile=xr.string_to_path(ctx.instance, profile),
                            suggested_bindings=[
                                xr.ActionSuggestedBinding(act, xr.string_to_path(ctx.instance, path))
                                for act, path in bindings
                            ],
                        ),
                    )
                except Exception as e:
                    debug("Could not suggest bindings for %s: %s", profile, e)

            frame_idx = [0]
            last_advance = [time.monotonic()]
            pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # accumulated pan
            base_distance = -molecule_offset_z
            distance = [base_distance]  # mutable for zoom
            orbit_rot_x = [0.0]
            orbit_rot_y = [0.0]
            ROTATE_DELTA = 0.26  # ~15 deg per press
            show_text = [True]
            last_toggle_text = [False]
            last_switch_mode = [False]
            stop_exit_thread = [False]

            def _exit_thread_fn():
                """Keep setting exit_render_loop so frame_loop sees it (it resets each iter)."""
                while not stop_exit_thread[0]:
                    if switch_to_desktop and switch_to_desktop[0]:
                        ctx.exit_render_loop = True
                    time.sleep(0.002)

            exit_thread = threading.Thread(target=_exit_thread_fn, daemon=True)
            exit_thread.start()

            cells_list = cells if cells else [None] * len(frames)
            metadata_list = metadata if metadata else [{}] * len(frames)
            if len(cells_list) < len(frames):
                cells_list.extend([None] * (len(frames) - len(cells_list)))

            def get_atoms():
                if not frames:
                    return [], None, None
                idx = frame_idx[0] % len(frames)
                now = time.monotonic()
                if len(frames) > 1 and (now - last_advance[0]) >= 1.0 / playback_fps:
                    frame_idx[0] = (frame_idx[0] + 1) % len(frames)
                    last_advance[0] = now
                f = frames[frame_idx[0] % len(frames)]
                if not f:
                    return [], None, None
                n = len(f)
                center = (
                    sum(a.x for a in f) / n,
                    sum(a.y for a in f) / n,
                    sum(a.z for a in f) / n,
                )
                cell = cells_list[frame_idx[0] % len(frames)]
                return center_and_scale(f, scale=scale), center, cell

            for frame_state in ctx.frame_loop():
                if ctx.exit_render_loop:
                    break

                # Sync and read controller state
                try:
                    xr.sync_actions(
                        ctx.session,
                        xr.ActionsSyncInfo(
                            active_action_sets=[
                                xr.ActiveActionSet(ctx.default_action_set, 0),
                            ],
                        ),
                    )
                    move_state_left = xr.get_action_state_vector2f(
                        ctx.session,
                        xr.ActionStateGetInfo(move_action, subaction_path=hand_left),
                    )
                    zoom_state_right = xr.get_action_state_vector2f(
                        ctx.session,
                        xr.ActionStateGetInfo(zoom_action, subaction_path=hand_right),
                    )
                    recenter_left = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(recenter_action, subaction_path=hand_left),
                    )
                    recenter_right = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(recenter_action, subaction_path=hand_right),
                    )
                    view_x = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(view_x_action, subaction_path=hand_left),
                    )
                    view_y = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(view_y_action, subaction_path=hand_left),
                    )
                    view_z = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(view_z_action, subaction_path=hand_left),
                    )
                    toggle_text_left = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(toggle_text_action, subaction_path=hand_left),
                    )
                    toggle_text_right = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(toggle_text_action, subaction_path=hand_right),
                    )
                    switch_mode_left = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(switch_mode_action, subaction_path=hand_left),
                    )
                    switch_mode_right = xr.get_action_state_boolean(
                        ctx.session,
                        xr.ActionStateGetInfo(switch_mode_action, subaction_path=hand_right),
                    )
                    toggle_now = toggle_text_left.current_state or toggle_text_right.current_state
                    if toggle_now and not last_toggle_text[0]:
                        show_text[0] = not show_text[0]
                    last_toggle_text[0] = toggle_now
                    switch_now = switch_mode_left.current_state or switch_mode_right.current_state
                    if switch_now and not last_switch_mode[0]:
                        if switch_to_desktop is not None:
                            switch_to_desktop[0] = True
                        ctx.exit_render_loop = True
                    last_switch_mode[0] = switch_now
                    if view_x.current_state:
                        orbit_rot_y[0] += ROTATE_DELTA
                    if view_y.current_state:
                        orbit_rot_x[0] += ROTATE_DELTA
                    if view_z.current_state:
                        orbit_rot_y[0] -= ROTATE_DELTA
                    if recenter_left.current_state or recenter_right.current_state:
                        pan_offset[0] = pan_offset[1] = 0.0
                        distance[0] = base_distance
                    if move_state_left.is_active:
                        speed = 0.6  # per-frame pan speed
                        pan_offset[0] += move_state_left.current_state.x * speed
                        pan_offset[1] += move_state_left.current_state.y * speed
                    if zoom_state_right.is_active:
                        zoom_speed = 1.2
                        distance[0] = max(0.1, min(500.0, distance[0] - zoom_state_right.current_state.y * zoom_speed))
                except Exception as e:
                    debug("Controller sync/read failed: %s", e)

                if not frame_state.should_render:
                    continue

                # Compute molecule position once per frame (in front of head center)
                molecule_world_pos = [None]
                dist = distance[0]
                # Ortho zoom: scale view by distance so zoom affects apparent size
                zoom_factor = dist / base_distance

                # Iterate over each eye view
                for view in ctx.view_loop(frame_state):
                    clear_screen()
                    setup_projection_vr(view.fov, near=0.05, far=500.0, zoom=zoom_factor)
                    setup_view_vr(view_matrix_from_posef(view.pose))

                    # Pointer: draw laser from controller aim pose (in reference space)
                    if pointer:
                        for ptr_space in (pointer_space_left, pointer_space_right):
                            try:
                                loc = xr.locate_space(ptr_space, ctx.space, frame_state.predicted_display_time)
                                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT and loc.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT:
                                    R = rotation_from_quaternionf(loc.pose.orientation)
                                    fwd = R @ np.array([0, 0, -1], dtype=np.float32)
                                    start = (loc.pose.position.x, loc.pose.position.y, loc.pose.position.z)
                                    draw_pointer_ray(start, tuple(fwd), length=3.0)
                            except Exception as e:
                                debug("Pointer locate failed: %s", e)

                    # Place molecule directly in front of view (head + distance * forward) + pan
                    if molecule_world_pos[0] is None:
                        R = rotation_from_quaternionf(view.pose.orientation)
                        forward = R @ np.array([0, 0, -1], dtype=np.float32)  # -Z = forward
                        right = R @ np.array([1, 0, 0], dtype=np.float32)
                        up = R @ np.array([0, 1, 0], dtype=np.float32)
                        pos = (
                            np.array([
                                view.pose.position.x,
                                view.pose.position.y,
                                view.pose.position.z,
                            ], dtype=np.float32)
                            + dist * forward
                            + pan_offset[0] * right
                            + pan_offset[1] * up
                        )
                        molecule_world_pos[0] = pos
                    glTranslatef(
                        float(molecule_world_pos[0][0]),
                        float(molecule_world_pos[0][1]),
                        float(molecule_world_pos[0][2]),
                    )
                    # Orbit rotation: X/Y/Z buttons add rotation around axes
                    glRotatef(orbit_rot_y[0] * 57.3, 0.0, 1.0, 0.0)
                    glRotatef(orbit_rot_x[0] * 57.3, 1.0, 0.0, 0.0)

                    atoms, center, cell = get_atoms()
                    if atoms:
                        if cell is not None:
                            draw_pbc_cell(cell, center, scale)
                        bonds = compute_bonds(atoms)
                        draw_molecule(atoms, bonds=bonds, atom_scale=atom_scale)
                        cam_pos = (view.pose.position.x, view.pose.position.y, view.pose.position.z)
                        mol_pos = (float(molecule_world_pos[0][0]), float(molecule_world_pos[0][1]), float(molecule_world_pos[0][2]))
                        if show_text[0]:
                            draw_annotations(atoms, bonds, cam_pos, scale=0.28, molecule_world_offset=mol_pos)
                    # Overlay: use current viewport (0,0 triggers glGetIntegerv in draw_overlay)
                    if show_text[0]:
                        meta = metadata_list[frame_idx[0] % len(metadata_list)] if metadata_list else {}
                        draw_overlay(0, 0, frame_idx[0] % len(frames), len(frames), meta)

            stop_exit_thread[0] = True
    except Exception:
        try:
            context_provider.destroy()
        except Exception:
            pass
        raise
