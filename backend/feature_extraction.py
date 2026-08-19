import numpy as np
from scipy.interpolate import interp1d

# Define the 7 pose landmarks we want to keep
# 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]

def extract_raw_landmarks(results):
    """Extracts raw landmarks from MediaPipe results."""
    # Pose: 33 landmarks by default, we extract them all first to compute centers
    if results.pose_landmarks:
        pose = np.array([[r.x, r.y, r.z] for r in results.pose_landmarks])
    else:
        pose = np.zeros((33, 3))

    if results.left_hand_landmarks:
        lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks])
    else:
        lh = None

    if results.right_hand_landmarks:
        rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks])
    else:
        rh = None

    return pose, lh, rh

def process_video(results_list, target_frames=50):
    """
    Processes a list of MediaPipe results into a fixed-length feature sequence.
    Returns: numpy array of shape (target_frames, 546), or None if sequence is invalid.
    """
    if not results_list:
        return None

    raw_sequence = [extract_raw_landmarks(res) for res in results_list]
    
    # 1. Forward/Backward Fill Missing Hands
    lh_valid_idx = [i for i, (p, lh, rh) in enumerate(raw_sequence) if lh is not None]
    rh_valid_idx = [i for i, (p, lh, rh) in enumerate(raw_sequence) if rh is not None]

    if len(lh_valid_idx) == 0 and len(rh_valid_idx) == 0:
        return None # Discard video only if BOTH hands are never detected
        
    # Left hand
    lh_filled = []
    if len(lh_valid_idx) == 0:
        # Hand entirely missing. Park it at the left shoulder.
        for p, _, _ in raw_sequence:
            lh_filled.append(np.tile(p[11], (21, 1)))
    else:
        last_lh = raw_sequence[lh_valid_idx[0]][1] # Backward fill first valid
        for i, (p, lh, rh) in enumerate(raw_sequence):
            if lh is not None:
                last_lh = lh
            lh_filled.append(last_lh)

    # Right hand
    rh_filled = []
    if len(rh_valid_idx) == 0:
        # Hand entirely missing. Park it at the right shoulder.
        for p, _, _ in raw_sequence:
            rh_filled.append(np.tile(p[12], (21, 1)))
    else:
        last_rh = raw_sequence[rh_valid_idx[0]][2] # Backward fill first valid
        for i, (p, lh, rh) in enumerate(raw_sequence):
            if rh is not None:
                last_rh = rh
            rh_filled.append(last_rh)

    # 2. Normalization
    features_sequence = []
    for i in range(len(raw_sequence)):
        pose = raw_sequence[i][0]
        lh = lh_filled[i]
        rh = rh_filled[i]

        # Body-level normalization
        l_sh = pose[11]
        r_sh = pose[12]
        body_center = (l_sh + r_sh) / 2.0
        body_scale = np.linalg.norm(l_sh - r_sh)
        if body_scale < 1e-4: body_scale = 1.0

        # Trim pose to 7 landmarks and normalize
        trimmed_pose = pose[POSE_INDICES]
        body_rel_pose = (trimmed_pose - body_center) / body_scale
        body_rel_lh = (lh - body_center) / body_scale
        body_rel_rh = (rh - body_center) / body_scale

        # Hand-level normalization
        # Left hand relative
        lh_wrist = lh[0]
        lh_mcp = lh[9]
        lh_scale = np.linalg.norm(lh_wrist - lh_mcp)
        if lh_scale < 1e-4: lh_scale = 1.0
        hand_rel_lh = (lh - lh_wrist) / lh_scale

        # Right hand relative
        rh_wrist = rh[0]
        rh_mcp = rh[9]
        rh_scale = np.linalg.norm(rh_wrist - rh_mcp)
        if rh_scale < 1e-4: rh_scale = 1.0
        hand_rel_rh = (rh - rh_wrist) / rh_scale

        # Concatenate position features (21*3 + 21*3 + 21*3 + 21*3 + 21*3 = 273)
        # Actually trimmed_pose is 7*3 = 21. lh is 21*3=63.
        # 21 + 63 + 63 + 63 + 63 = 273 features
        frame_features = np.concatenate([
            body_rel_pose.flatten(),
            body_rel_lh.flatten(),
            body_rel_rh.flatten(),
            hand_rel_lh.flatten(),
            hand_rel_rh.flatten()
        ])
        features_sequence.append(frame_features)

    features_sequence = np.array(features_sequence)

    # 3. Motion Features (Velocity)
    velocities = np.zeros_like(features_sequence)
    if len(features_sequence) > 1:
        velocities[1:] = features_sequence[1:] - features_sequence[:-1]
        
    # 3.5 Trim Idle Frames based on motion energy (velocity magnitude)
    energies = np.sum(np.abs(velocities), axis=1)
    if len(energies) > 3:
        smoothed = np.convolve(energies, np.ones(3)/3, mode='same')
    else:
        smoothed = energies
        
    if len(smoothed) > 0:
        peak = np.max(smoothed)
        threshold = 0.15 * peak # 15% of peak velocity
        active_indices = np.where(smoothed > threshold)[0]
        
        if len(active_indices) > 0:
            start_idx = max(0, active_indices[0] - 1) # Add 1 frame padding
            end_idx = min(len(features_sequence) - 1, active_indices[-1] + 1)
            features_sequence = features_sequence[start_idx:end_idx+1]
            velocities = velocities[start_idx:end_idx+1]
    
    # Concatenate positions and velocities (273 + 273 = 546)
    full_sequence = np.concatenate([features_sequence, velocities], axis=1)

    # 4. Temporal Resampling (Linear Interpolation) to target_frames (30)
    seq_len = len(full_sequence)
    if seq_len == target_frames:
        return full_sequence
    
    if seq_len == 1:
        # If only 1 frame, just repeat it 30 times
        return np.repeat(full_sequence, target_frames, axis=0)
    
    # Original time indices
    old_indices = np.linspace(0, 1, seq_len)
    new_indices = np.linspace(0, 1, target_frames)
    
    interpolator = interp1d(old_indices, full_sequence, axis=0, kind='linear')
    resampled_sequence = interpolator(new_indices)

    return resampled_sequence
