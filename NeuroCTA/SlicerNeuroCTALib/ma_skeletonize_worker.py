# ma_skeletonize_worker.py
import sys
import json
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt

def compute_soam(coords, voxel_spacing):
    """
    Compute the Sum of Angles Metric (SOAM) for a sequence of 3D coordinates.
    Coords are in voxel (ZYX) space; voxel_spacing converts to physical units.
    Returns SOAM in radians per unit length.

    Tortuosity - Sum of Angle Metric (SOAM) = length-weighted average across all branches in segment (Bullitt et al., 2003)

    """
    if len(coords) < 4:
        return 0.0

    # Convert to physical space
    pts = coords * np.array(voxel_spacing)

    total_angle = 0.0
    valid_points = 0

    for k in range(1, len(pts) - 2):
        T1 = pts[k] - pts[k - 1]
        T2 = pts[k + 1] - pts[k]
        T3 = pts[k + 2] - pts[k + 1]

        n1 = np.linalg.norm(T1)
        n2 = np.linalg.norm(T2)
        n3 = np.linalg.norm(T3)

        if n1 < 1e-6 or n2 < 1e-6 or n3 < 1e-6:
            continue

        T1n = T1 / n1
        T2n = T2 / n2
        T3n = T3 / n3

        # In-plane angle at Pk
        dot_ip = np.clip(T1n @ T2n, -1.0, 1.0)
        IP = np.arccos(dot_ip)

        # Torsional angle at Pk
        cross1 = np.cross(T1n, T2n)
        cross2 = np.cross(T2n, T3n)
        nc1 = np.linalg.norm(cross1)
        nc2 = np.linalg.norm(cross2)

        if nc1 < 1e-6 or nc2 < 1e-6:
            TP = 0.0
        else:
            cross1n = cross1 / nc1
            cross2n = cross2 / nc2
            dot_tp = np.clip(cross1n @ cross2n, -1.0, 1.0)
            TP = np.arccos(dot_tp)
            # Set to 0 at inflection points (TP == pi)
            if np.isclose(TP, np.pi):
                TP = 0.0

        # Combined angle at Pk
        total_angle += np.sqrt(IP**2 + TP**2)
        valid_points += 1

    if valid_points == 0:
        return 0.0

    # Normalize by total curve length
    deltas = np.diff(pts, axis=0)
    total_length = np.sum(np.linalg.norm(deltas, axis=1))

    if total_length < 1e-6:
        return 0.0

    return total_angle / total_length

def _compute_features(label, seg_stats, full_skel_obj, full_dist_transform, skel_mask, segment_mask, bp_coords, ep_coords, voxel_spacing):
        
    features = {}

    if len(seg_stats) == 0:
        return features
    
    features["length"] = float(seg_stats["branch-distance"].sum())
    features["n_branches"] = int(len(seg_stats))
    features["n_branch_points"] = int((seg_stats["branch-type"] == 2).sum())
        
    skeleton_radii = full_dist_transform[skel_mask]
    features["mean_radius"] = float(skeleton_radii.mean())
    features["min_radius"] = float(skeleton_radii.min())
    features["max_radius"] = float(skeleton_radii.max())

    # Tortuosity - Distance Metric (DM) = branch distance / euclidean distance (Bullitt et al., 2003)
    features["tortuosity_dm"] = float(
        seg_stats["branch-distance"].sum() / seg_stats["euclidean-distance"].sum()
    )

    # SOAM — length-weighted average across branches
    soam_total, length_total = 0.0, 0.0

    # Filter junction-to-junction stubs - these are artifacts of the
    # skeleton topology at branch points, not real vessel segments.
    # Threshold: shorter than 2x mean voxel spacing is usually a stub.
    min_meaningful_length = 2.0 * np.mean(voxel_spacing)
    
    for row_idx in seg_stats.index:
        path = full_skel_obj.path_coordinates(int(row_idx)).astype(float)
        branch_type = int(seg_stats.loc[row_idx, "branch-type"])
        branch_length = float(seg_stats.loc[row_idx, "branch-distance"])

        # Skip branches too short to compute any angle (need at least 4 points)
        # and junction-to-junction stubs (branch-type == 2 and very short)
        if len(path) < 4:
            continue
        if branch_type == 2 and branch_length < min_meaningful_length:
            continue
        
        soam_val = compute_soam(path, voxel_spacing)
        soam_total += soam_val * branch_length
        length_total += branch_length

    features["tortuosity_soam"] = soam_total / length_total if length_total > 0 else 0.0

    # branch points for this segment
    if len(bp_coords) > 0:
        bp_mask = segment_mask[bp_coords[:, 0], bp_coords[:, 1], bp_coords[:, 2]]
        features["branch_point_coords"] = bp_coords[bp_mask].tolist()
    else:
        features["branch_point_coords"] = []

    # end points for this segment
    if len(ep_coords) > 0:
        ep_mask = segment_mask[ep_coords[:, 0], ep_coords[:, 1], ep_coords[:, 2]]
        features["end_point_coords"] = ep_coords[ep_mask].tolist()
    else:
        features["end_point_coords"] = []   

    return features 


def main():
    args = json.loads(sys.argv[1])
    volume = np.load(args["volume_path"])
    labels_info = json.loads(open(args["labels_path"]).read())
    
    total_progress_steps = len(labels_info) + 2

    voxel_spacing = args.get("voxel_spacing", [1.0, 1.0, 1.0])

    full_skeleton = skeletonize((volume > 0).astype(np.uint8))

    # Updates progress bar (see _onMedialAxisProgressInfo in SkeletonizationLogic.py)
    print(f"PROGRESS:{1}:{total_progress_steps}\t[{1}/{total_progress_steps}] Skeletonization", flush=True)

    # Compute distance transform for all segments at once to avoid redundant calculations
    full_dist_transform = distance_transform_edt(volume > 0, sampling=voxel_spacing)
    full_skel_obj = Skeleton(full_skeleton, spacing=voxel_spacing)
    full_skel_stats = summarize(skel=full_skel_obj, separator='-')

    # Updates progress bar (see _onMedialAxisProgressInfo in SkeletonizationLogic.py)
    print(f"PROGRESS:{2}:{total_progress_steps}\t[{2}/{total_progress_steps}] Distance Transform", flush=True)

    if len(full_skel_stats) == 0:
        with open(args["output_path"], "w") as f:
            json.dump({}, f)
        return

    # Get all branch point and end points
    bp_coords = full_skel_obj.coordinates[np.where(full_skel_obj.degrees > 2)[0]]
    ep_coords = full_skel_obj.coordinates[np.where(full_skel_obj.degrees == 1)[0]]
    bp_coords = np.array(bp_coords.tolist(), dtype=int)
    ep_coords = np.array(ep_coords.tolist(), dtype=int)

    # Assigns each branch to a segment (vessel class) based on the label at the midpoint of the branch
    branch_coords = np.array([
        full_skel_obj.path_coordinates(i)[len(full_skel_obj.path_coordinates(i))//2].astype(int)
        for i in range(len(full_skel_stats))
    ])

    if len(branch_coords) == 0:
        branch_labels = np.array([], dtype=int)
    else:
        branch_labels = volume[branch_coords[:, 0], branch_coords[:, 1], branch_coords[:, 2]]

    full_skel_stats["segment_label"] = branch_labels

    # Iterate through each segment to extract skeleton coordinates, features, and branch/end points
    results = {}
    for i, label_info in enumerate(labels_info):
        label = label_info["label"]
        segment_name = label_info["name"]
        segment_mask = (volume == label)
        skel_mask = full_skeleton & segment_mask
        skel_coords = np.argwhere(skel_mask)

        seg_stats = full_skel_stats[full_skel_stats["segment_label"] == label]
        features = _compute_features(
            label, seg_stats, full_skel_obj, full_dist_transform, skel_mask, segment_mask, bp_coords, ep_coords, voxel_spacing
        )
        
        results[segment_name] = {
            "coords":   skel_coords.tolist(),
            "color":    label_info["color"],
            "features": features
        }

        print(f"PROGRESS:{i+3}:{total_progress_steps}\t[{i+3}/{total_progress_steps}] {segment_name} - {len(skel_coords)} points", flush=True)
    
    with open(args["output_path"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
