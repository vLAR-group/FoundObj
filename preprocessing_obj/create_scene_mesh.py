"""Step 1: construct a randomized indoor scene around each main object.

Each scene = a random floor, an optional wall, the main object (randomly rotated
and placed against the wall / on the floor), and a few clutter objects placed by
physically plausible rules (around / above / on-surface / below / on-wall). The
whole scene is finally normalized into a unit cube.
"""
import trimesh, random, json, os, argparse, sys
import numpy as np
from trimesh.collision import CollisionManager
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from preprocessing_obj.dataset_utils import discover_datasets


def export_components(scene, output_base_path):
    """Export every geometry node as <name>.ply, plus the merged full_scene.ply."""
    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        scene.geometry[geom_name].export(f"{output_base_path}/{node_name}.ply")
    scene.export(f"{output_base_path}/full_scene.ply")


def generate_random_floor():
    """A square floor in the XY plane with a random side length (1.4-1.5)."""
    half_size = random.uniform(1.4, 1.5) / 2
    lo, hi = -half_size, half_size
    floor_vertices = np.array([[lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo]])
    floor = trimesh.Trimesh(vertices=floor_vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]))
    floor.visual.face_colors = [200, 200, 200, 255]
    return floor


def generate_random_wall(floor, rand_v=0.8):
    """With probability rand_v, return a random wall on the floor's -x edge; else None."""
    if random.random() > rand_v:
        return None
    floor_min_z = floor.bounds[0][2]
    floor_min_x = floor.bounds[0][0]
    wall_height = random.uniform(1.2, 1.5)
    half_width = random.uniform(0.5, 1.5) / 2          # wall spans +/- half_width in y
    wall_vertices = np.array([
        [floor_min_x, -half_width, floor_min_z],
        [floor_min_x, half_width, floor_min_z],
        [floor_min_x, half_width, floor_min_z + wall_height],
        [floor_min_x, -half_width, floor_min_z + wall_height]])
    wall = trimesh.Trimesh(vertices=wall_vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]))
    wall.visual.face_colors = [150, 150, 200, 255]
    return wall


def normalize_scene(scene):
    """Translate to the bbox center and scale so the largest extent is 1."""
    bbox = scene.bounds
    center = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    for geom in scene.geometry.values():
        geom.apply_translation(-center)
        geom.apply_scale(1.0 / scale)
    return scene, center, scale


def random_rotate_mesh(mesh):
    """Rotate the mesh by a random angle about the z axis."""
    angle_z = random.uniform(0, 2 * np.pi)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_z, [0, 0, 1])
    mesh.apply_transform(rotation_matrix)
    return mesh, rotation_matrix

def place_mesh(mesh, floor, wall):
    """Put the main object on the floor, flush against the wall (+x of the wall),
    at a random y within the floor/wall overlap. Falls back to floor center if no wall."""
    if wall is None:
        return place_mesh_on_floor_center(mesh, floor)
    mesh_bounds = mesh.bounds
    mesh_min_z, mesh_min_x = mesh_bounds[0][2], mesh_bounds[0][0]
    floor_bounds, wall_bounds = floor.bounds, wall.bounds
    floor_z = floor_bounds[1][2]
    floor_min_y, floor_max_y = floor_bounds[0][1], floor_bounds[1][1]
    wall_x = wall_bounds[1][0]                       # touch the wall's +x face
    wall_min_y, wall_max_y = wall_bounds[0][1], wall_bounds[1][1]

    translation_z = floor_z - mesh_min_z
    translation_x = wall_x - mesh_min_x

    mesh_y_size = mesh_bounds[1][1] - mesh_bounds[0][1]
    mesh_center_y = (mesh_bounds[1][1] + mesh_bounds[0][1]) / 2
    # random y within the floor/wall overlap, keeping the object fully inside
    min_y_center = max(floor_min_y, wall_min_y) + mesh_y_size / 2
    max_y_center = min(floor_max_y, wall_max_y) - mesh_y_size / 2
    if max_y_center > min_y_center:
        target_y = random.uniform(min_y_center, max_y_center)
    else:
        target_y = (floor_min_y + floor_max_y) / 2
    translation_y = target_y - mesh_center_y

    mesh.apply_translation([translation_x, translation_y, translation_z])
    return mesh, [float(translation_x), float(translation_y), float(translation_z)]


def place_mesh_on_floor_center(mesh, floor):
    """Put the main object at the center of the floor (used when there is no wall)."""
    mesh_bounds = mesh.bounds
    mesh_min_z = mesh_bounds[0][2]
    mesh_center_x = (mesh_bounds[1][0] + mesh_bounds[0][0]) / 2
    mesh_center_y = (mesh_bounds[1][1] + mesh_bounds[0][1]) / 2

    floor_bounds = floor.bounds
    floor_z = floor_bounds[1][2]
    floor_center_x = (floor_bounds[1][0] + floor_bounds[0][0]) / 2
    floor_center_y = (floor_bounds[1][1] + floor_bounds[0][1]) / 2

    translation = [floor_center_x - mesh_center_x,
                   floor_center_y - mesh_center_y,
                   floor_z - mesh_min_z]
    mesh.apply_translation(translation)
    return mesh, [float(t) for t in translation]


def _pick_other_object(scale_config, other_objects_path, rotate=True):
    """Pick a random instance from one of the scale_config categories present in the
    clutter library, load it, apply a random per-category scale and (optionally) a
    random z-rotation. Returns (mesh, category) or (None, None)."""
    other_path = Path(other_objects_path)
    if not other_path.exists():
        print(f"Warning: augobj path not found: {other_objects_path}")
        return None, None
    categories = [d.name for d in other_path.iterdir() if d.is_dir() and d.name in scale_config]
    if not categories:
        return None, None
    category = random.choice(categories)
    mesh_files = [sub / "mesh.ply" for sub in (other_path / category).iterdir()
                  if sub.is_dir() and (sub / "mesh.ply").exists()]
    if not mesh_files:
        return None, None
    other_mesh = trimesh.load(str(random.choice(mesh_files)))
    lo, hi = scale_config[category]
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= random.uniform(lo, hi)
    other_mesh.apply_transform(scale_matrix)
    if rotate:
        angle_z = random.uniform(0, 2 * np.pi)
        other_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_z, [0, 0, 1]))
    return other_mesh, category


def _collides(mesh_a, mesh_b):
    cm = CollisionManager()
    cm.add_object('a', mesh_a)
    cm.add_object('b', mesh_b)
    return cm.in_collision_internal()


def load_and_place_other_object_around(mesh, floor, wall, other_objects_path):
    """Place a clutter object on the floor next to the main object (its +x side)."""
    scale_config = {'table': (0.4, 0.6), 'sofa': (1.0, 1.2), 'shelf': (1, 1.2),
                    'back chair': (0.5, 0.7), 'backless chair': (0.4, 0.6), 'bed': (1.0, 1.4)}
    other_mesh, category = _pick_other_object(scale_config, other_objects_path)
    if other_mesh is None:
        return None, None

    mesh_bounds, other_bounds = mesh.bounds, other_mesh.bounds
    mesh_center_y = (mesh_bounds[0][1] + mesh_bounds[1][1]) / 2
    other_center_y = (other_bounds[0][1] + other_bounds[1][1]) / 2
    # bottom on the floor, left edge at the main object's right edge, y aligned to center
    other_mesh.apply_translation([mesh_bounds[1][0] - other_bounds[0][0],
                                  mesh_center_y - other_center_y,
                                  floor.bounds[1][2] - other_bounds[0][2]])
    if _collides(mesh, other_mesh):
        print(f"Collision for {category} (around), skipping")
        return None, None
    return other_mesh, other_mesh.bounds


def load_and_place_other_object_above(mesh, floor, wall, other_objects_path):
    """Hang a ceiling object above the main object, its top flush with the ceiling."""
    scale_config = {'ceiling lamp': (0.5, 0.6)}
    other_mesh, category = _pick_other_object(scale_config, other_objects_path)
    if other_mesh is None:
        return None, None

    mesh_bounds, other_bounds = mesh.bounds, other_mesh.bounds
    mesh_center_x = (mesh_bounds[1][0] + mesh_bounds[0][0]) / 2
    mesh_center_y = (mesh_bounds[1][1] + mesh_bounds[0][1]) / 2
    other_center_x = (other_bounds[1][0] + other_bounds[0][0]) / 2
    other_center_y = (other_bounds[1][1] + other_bounds[0][1]) / 2
    other_height = other_bounds[1][2] - other_bounds[0][2]

    ceiling_height = wall.bounds[1][2] if wall is not None else mesh_bounds[1][2] + 0.5
    base_z = ceiling_height - other_height
    # small random xy offset around the main object's center
    mesh_center_x += random.uniform(-0.2, 0.2)
    mesh_center_y += random.uniform(-0.2, 0.2)
    other_mesh.apply_translation([mesh_center_x - other_center_x,
                                  mesh_center_y - other_center_y,
                                  base_z - other_bounds[0][2]])
    if _collides(mesh, other_mesh):
        print(f"Collision for {category} (above), skipping")
        return None, None
    return other_mesh, other_mesh.bounds


def load_and_place_other_object_on_surface(mesh, floor, wall, other_objects_path):
    """Place a small object (pillow/lamp/cup) on an upward-facing surface of the main object."""
    scale_config = {'pillow': (0.3, 0.4), 'table lamp': (0.3, 0.4), 'cup': (0.15, 0.25)}
    other_mesh, category = _pick_other_object(scale_config, other_objects_path)
    if other_mesh is None:
        return None, None

    # find a target point: prefer a random upward-facing sampled surface point
    if hasattr(mesh, 'sample') and hasattr(mesh, 'face_normals'):
        surface_points, face_indices = mesh.sample(1000, return_index=True)
        surface_normals = mesh.face_normals[face_indices]
        up_mask = surface_normals[:, 2] > 0.7
        if np.any(up_mask):
            up_points, up_normals = surface_points[up_mask], surface_normals[up_mask]
            idx = random.randint(0, len(up_points) - 1)
            target_point, surface_normal = up_points[idx], up_normals[idx]
        else:
            idx = np.argmax(surface_points[:, 2])
            target_point, surface_normal = surface_points[idx], surface_normals[idx]
    else:
        idx = np.argmax(mesh.vertices[:, 2])
        target_point, surface_normal = mesh.vertices[idx], np.array([0, 0, 1])

    other_bounds = other_mesh.bounds
    other_bottom_center = np.array([(other_bounds[1][0] + other_bounds[0][0]) / 2,
                                    (other_bounds[1][1] + other_bounds[0][1]) / 2,
                                    other_bounds[0][2]])
    translation = target_point - other_bottom_center
    if surface_normal[2] < 0.9:           # not horizontal: nudge along the normal to avoid clipping
        translation = translation + surface_normal * 0.02
    other_mesh.apply_translation(translation)
    return other_mesh, other_mesh.bounds


def load_and_place_other_object_below(mesh, floor, wall, other_objects_path):
    """Place an object on the floor under a downward-facing surface of the main object."""
    scale_config = {'backless chair': (0.35, 0.5)}
    other_mesh, category = _pick_other_object(scale_config, other_objects_path)
    if other_mesh is None:
        return None, None

    mesh_bounds = mesh.bounds
    mesh_center = np.array([(mesh_bounds[1][0] + mesh_bounds[0][0]) / 2,
                            (mesh_bounds[1][1] + mesh_bounds[0][1]) / 2, mesh_bounds[0][2]])
    # target xy: a random downward-facing sampled point, else the main object's bottom center
    target_point = mesh_center
    if hasattr(mesh, 'sample') and hasattr(mesh, 'face_normals'):
        surface_points, face_indices = mesh.sample(2000, return_index=True)
        down_mask = mesh.face_normals[face_indices][:, 2] < -0.7
        if np.any(down_mask):
            down_points = surface_points[down_mask]
            target_point = down_points[random.randint(0, len(down_points) - 1)]

    other_bounds = other_mesh.bounds
    other_center_x = (other_bounds[1][0] + other_bounds[0][0]) / 2
    other_center_y = (other_bounds[1][1] + other_bounds[0][1]) / 2
    other_mesh.apply_translation([target_point[0] - other_center_x,
                                  target_point[1] - other_center_y,
                                  floor.bounds[1][2] - other_bounds[0][2]])
    if _collides(mesh, other_mesh):
        print(f"Collision for {category} (below), skipping")
        return None, None
    return other_mesh, other_mesh.bounds


def load_and_place_other_object_on_wall(mesh, floor, wall, other_objects_path):
    """Hang a flat object (e.g. picture) on the wall, above the main object."""
    scale_config = {'picture': (0.3, 0.5)}
    if wall is None:
        return None, None
    other_mesh, category = _pick_other_object(scale_config, other_objects_path, rotate=False)
    if other_mesh is None:
        return None, None

    # rotate so the thinnest dimension is along x (the wall's normal)
    x_size, y_size, z_size = other_mesh.bounds[1] - other_mesh.bounds[0]
    min_size = min(x_size, y_size, z_size)
    if min_size == y_size:
        other_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    elif min_size == z_size:
        other_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    other_bounds = other_mesh.bounds
    other_center_x = (other_bounds[1][0] + other_bounds[0][0]) / 2
    other_center_y = (other_bounds[1][1] + other_bounds[0][1]) / 2
    other_center_z = (other_bounds[1][2] + other_bounds[0][2]) / 2
    mesh_bounds = mesh.bounds
    mesh_center_y = (mesh_bounds[1][1] + mesh_bounds[0][1]) / 2
    wall_x, wall_min_y, wall_max_y, wall_max_z = (wall.bounds[0][0], wall.bounds[0][1],
                                                  wall.bounds[1][1], wall.bounds[1][2])

    # z: above the main object, random offset, capped by the wall height
    target_z = mesh_bounds[1][2] + random.uniform(0.2, 0.8)
    target_z = min(target_z, wall_max_z - (other_bounds[1][2] - other_bounds[0][2]) / 2)
    # x: embed 70% of its depth into the wall
    target_x = wall_x + (other_bounds[1][0] - other_bounds[0][0]) * 0.7 / 2
    # y: biased toward the main object's center, within the wall
    other_y_size = other_bounds[1][1] - other_bounds[0][1]
    min_y_center, max_y_center = wall_min_y + other_y_size / 2, wall_max_y - other_y_size / 2
    if max_y_center > min_y_center:
        bias = random.uniform(-0.3, 0.3) * (max_y_center - min_y_center)
        target_y = max(min_y_center, min(max_y_center, mesh_center_y + bias))
    else:
        target_y = (wall_min_y + wall_max_y) / 2

    other_mesh.apply_translation([target_x - other_center_x,
                                  target_y - other_center_y,
                                  target_z - other_center_z])
    return other_mesh, other_mesh.bounds


def add_random_other_objects(scene, mesh, floor, wall, other_objects_path, prob=0.7):
    """With probability prob, add 1-5 clutter objects via randomly chosen placement rules."""
    if random.random() > prob:
        return scene, []

    placement_methods = [
        ('around', load_and_place_other_object_around),
        ('above', load_and_place_other_object_above),
        ('surface', load_and_place_other_object_on_surface),
        ('below', load_and_place_other_object_below),
        ('wall', load_and_place_other_object_on_wall)]
    selected_methods = random.sample(placement_methods, random.randint(1, 5))

    added_objects_info = []
    for method_name, method_func in selected_methods:
        other_mesh, other_bounds = method_func(mesh, floor, wall, other_objects_path)
        if other_mesh is not None:
            node_name = f"other_object_{method_name}"
            scene.add_geometry(other_mesh, node_name=node_name)
            added_objects_info.append((node_name, other_bounds))
        else:
            print(f"Failed to add object using {method_name} placement")
    return scene, added_objects_info


def generate_augmented_meshes(input_mesh_path, other_objects_path, output_dir):
    """Build one randomized scene around input_mesh_path and write it to output_dir."""
    output_path = Path(output_dir)
    mesh = trimesh.load(input_mesh_path)
    floor = generate_random_floor()
    wall = generate_random_wall(floor)
    mesh, _ = random_rotate_mesh(mesh)
    mesh, _ = place_mesh(mesh, floor, wall)

    scene = trimesh.Scene()
    scene.add_geometry(floor, node_name='floor')
    if wall is not None:
        scene.add_geometry(wall, node_name='wall')
    scene.add_geometry(mesh, node_name='object')

    scene, added_objects_info = add_random_other_objects(scene, mesh, floor, wall, other_objects_path, prob=0.7)
    added_objects_info.append(('object', mesh.bounds))     # bounds before normalization

    # normalize the whole scene, then record each object's normalized bbox center
    scene, center, scale = normalize_scene(scene)
    normalized_objects_info = {}
    for node_name, bounds in added_objects_info:
        normalized_bounds = (bounds - center) / scale
        bbox_center = (normalized_bounds[0] + normalized_bounds[1]) / 2
        normalized_objects_info[node_name] = bbox_center.tolist()

    export_components(scene, str(output_path))
    with open(f"{output_path}/center_loc.json", 'w') as f:
        json.dump(normalized_objects_info, f, indent=2)


def process_dataset(input_base_dir, output_base_dir, other_objects_path, num_variations):
    os.makedirs(output_base_dir, exist_ok=True)
    model_folders = [d for d in sorted(os.listdir(input_base_dir))
                     if os.path.exists(os.path.join(input_base_dir, d, "mesh.ply"))]
    for idx, model_folder in enumerate(model_folders, 1):
        mesh_file = os.path.join(input_base_dir, model_folder, "mesh.ply")
        print(f"Processing model: {model_folder}")
        for i in range(num_variations):
            output_dir = os.path.join(output_base_dir, f"{model_folder}_{i}")
            os.makedirs(output_dir, exist_ok=True)
            generate_augmented_meshes(mesh_file, other_objects_path, output_dir)
        print(f"Completed ({idx}/{len(model_folders)} models)")


NUM_VARIATIONS = 5    # scene variations generated per object


def main():
    parser = argparse.ArgumentParser(description="Construct randomized indoor scenes around target objects")
    parser.add_argument("--data_root", default="data/objects")
    parser.add_argument("--augobj_dir", default="data/objects/Others_augobj",
                        help="Clutter library (the downloaded Others_augobj folder)")
    args = parser.parse_args()

    for name in discover_datasets(args.data_root):
        print(f"\n=== dataset: {name} ===")
        process_dataset(
            os.path.join(args.data_root, name, "renders"),
            os.path.join(args.data_root, name, "syn_mesh"),
            args.augobj_dir, NUM_VARIATIONS)


if __name__ == "__main__":
    main()