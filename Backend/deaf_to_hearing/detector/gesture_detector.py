import numpy as np

# --- TEST VALUES (Calibration needed for live testing) ---
TEST_HOOK_ANGLE_MIN = 100
TEST_HOOK_ANGLE_MAX = 140
TEST_PALM_Z_TOLERANCE = 0.05
DEBUG_PRINT_METRICS = False
# ---------------------------------------------------------

class GestureDetector:
    def __init__(self):
        pass

    def _get_distance_2d(self, pt1, pt2):
        """Calculate stable 2D Euclidean distance in the XY plane."""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def _get_angle(self, a, b, c):
        """Calculate angle at joint b given points a, b, c in 2D (in degrees)."""
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _extract_features(self, hand_landmarks):
        pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
        pts_z = [getattr(lm, 'z', 0.0) for lm in hand_landmarks.landmark]
        wrist = pts[0]
        
        palm_width = self._get_distance_2d(pts[5], pts[17])
        palm_height = self._get_distance_2d(pts[0], pts[9])
        scale = max(palm_width, palm_height * 0.55)
        if scale < 1e-4:
            scale = 1e-4

        fingers_extended = {}
        finger_indices = {
            'index': (5, 6, 8),
            'middle': (9, 10, 12),
            'ring': (13, 14, 16),
            'pinky': (17, 18, 20)
        }
        
        for name, (mcp_idx, pip_idx, tip_idx) in finger_indices.items():
            d_tip = self._get_distance_2d(pts[tip_idx], wrist)
            d_pip = self._get_distance_2d(pts[pip_idx], wrist)
            d_mcp = self._get_distance_2d(pts[mcp_idx], wrist)
            is_extended = d_tip > d_pip * 1.05 and d_tip > d_mcp * 1.15
            fingers_extended[name] = is_extended

        t_tip = pts[4]
        i_mcp = pts[5]
        p_mcp = pts[17]
        
        thumb_index_dist = self._get_distance_2d(t_tip, i_mcp) / scale
        thumb_pinky_dist = self._get_distance_2d(t_tip, p_mcp) / scale
        thumb_extended = thumb_index_dist > 0.38 and thumb_pinky_dist > 0.75

        i_tip = pts[8]
        m_tip = pts[12]
        r_tip = pts[16]
        p_tip = pts[20]

        d_thumb_index = self._get_distance_2d(t_tip, i_tip) / scale
        d_thumb_middle = self._get_distance_2d(t_tip, m_tip) / scale
        d_thumb_ring = self._get_distance_2d(t_tip, r_tip) / scale
        d_thumb_pinky = self._get_distance_2d(t_tip, p_tip) / scale

        avg_all_pinched = (d_thumb_index + d_thumb_middle + d_thumb_ring + d_thumb_pinky) / 4.0

        tip_spread = (self._get_distance_2d(i_tip, m_tip) + 
                      self._get_distance_2d(m_tip, r_tip) + 
                      self._get_distance_2d(r_tip, p_tip)) / scale

        index_pip_angle = self._get_angle(pts[5], pts[6], pts[8])
        palm_z_diff = abs(pts_z[5] - pts_z[17])
        thumb_pip_angle = self._get_angle(pts[1], pts[2], pts[4])

        return {
            'pts': pts,
            'pts_z': pts_z,
            'scale': scale,
            'fingers_extended': fingers_extended,
            'thumb_extended': thumb_extended,
            'thumb_index_dist': thumb_index_dist,
            'thumb_pinky_dist': thumb_pinky_dist,
            'd_thumb_index': d_thumb_index,
            'd_thumb_middle': d_thumb_middle,
            'd_thumb_ring': d_thumb_ring,
            'd_thumb_pinky': d_thumb_pinky,
            'avg_all_pinched': avg_all_pinched,
            'tip_spread': tip_spread,
            'index_pip_angle': index_pip_angle,
            'palm_z_diff': palm_z_diff,
            'thumb_pip_angle': thumb_pip_angle
        }

    def _detect_numbers(self, features):
        fingers_extended = features['fingers_extended']
        thumb_extended = features['thumb_extended']
        d_thumb_index = features['d_thumb_index']
        avg_all_pinched = features['avg_all_pinched']
        pts = features['pts']
        palm_z_diff = features['palm_z_diff']
        index_pip_angle = features['index_pip_angle']

        folded_count = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not fingers_extended[name]])
        
        # 0: Fist / 'O'
        if folded_count == 4:
            if not thumb_extended or d_thumb_index < 0.5 or avg_all_pinched < 0.6:
                return "0", 0.98
            
        # 5: All 5 fingers extended
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky'] and thumb_extended:
            return "5", 0.98
            
        # 4: Index, Middle, Ring, Pinky extended, Thumb folded
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky'] and not thumb_extended:
            return "4", 0.98
            
        # 3: Index, Middle, Ring extended, Pinky & Thumb folded
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and not fingers_extended['pinky'] and not thumb_extended:
            return "3", 0.98
            
        # 2: Index and Middle extended, others folded
        if fingers_extended['index'] and fingers_extended['middle'] and not fingers_extended['ring'] and not fingers_extended['pinky'] and not thumb_extended:
            return "2", 0.98
            
        # 1: Index extended, others folded
        if fingers_extended['index'] and not fingers_extended['middle'] and not fingers_extended['ring'] and not fingers_extended['pinky'] and not thumb_extended:
            return "1", 0.98
            
        # 6: Thumb extended, others folded (with Z-depth check for front-facing palm)
        if thumb_extended and folded_count == 4:
            if pts[4][1] < pts[2][1] - 0.05:
                if palm_z_diff < TEST_PALM_Z_TOLERANCE:
                    return "6", 0.95
                
        # 7: Index and Thumb hooked/curved (C-shape), others folded
        if folded_count >= 3 and not fingers_extended['middle'] and not fingers_extended['ring'] and not fingers_extended['pinky']:
            if TEST_HOOK_ANGLE_MIN <= index_pip_angle <= TEST_HOOK_ANGLE_MAX:
                return "7", 0.95
                
        # 8: Thumb, Index, Middle extended, others folded
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and not fingers_extended['ring'] and not fingers_extended['pinky']:
            return "8", 0.98
            
        # 9: Middle, Ring, Pinky extended, Index on Thumb (pinched)
        if fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']:
            if d_thumb_index < 0.35:
                return "9", 0.98
        
        return "Unknown", 0.0

    def _detect_words(self, features):
        fingers_extended = features['fingers_extended']
        thumb_extended = features['thumb_extended']
        avg_all_pinched = features['avg_all_pinched']
        d_thumb_pinky = features['d_thumb_pinky']
        d_thumb_index = features['d_thumb_index']
        d_thumb_middle = features['d_thumb_middle']
        thumb_pinky_dist = features['thumb_pinky_dist']
        scale = features['scale']
        pts = features['pts']
        tip_spread = features['tip_spread']

        # --- Pinched & Touch Gestures ---
        if avg_all_pinched < 0.48 and d_thumb_pinky < 0.50:
            return "Food", 0.98
        if d_thumb_index < 0.38 and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']:
            return "OK", 0.98
        if d_thumb_middle < 0.35 and fingers_extended['index'] and fingers_extended['ring'] and fingers_extended['pinky']:
            return "Money", 0.98
        if d_thumb_pinky < 0.30 and fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring']:
            return "Attention", 0.98

        # --- Special Multi-Finger Combo Shapes ---
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['pinky'] and not fingers_extended['ring']:
            return "Read", 0.98
        if thumb_extended and not fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']:
            return "Emergency", 0.98
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and not fingers_extended['ring'] and fingers_extended['pinky']:
            return "Question", 0.98
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and not fingers_extended['pinky']:
            return "Good Morning", 0.98

        # --- Flat Open Hand Expressions: Hello, Stop, Please ---
        all_four_extended = fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']
        if all_four_extended:
            if thumb_pinky_dist <= 0.78:
                return "Please", 0.98
            else:
                i_tip = pts[8]
                p_tip = pts[20]
                tip_width = self._get_distance_2d(i_tip, p_tip) / scale
                if tip_width > 1.25 or tip_spread > 0.85:
                    return "Hello", 0.95
                else:
                    return "Stop", 0.95

        # --- 3-Finger and 2-Finger Shapes ---
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and not (fingers_extended['ring'] or fingers_extended['pinky']):
            return "How are you", 0.98
        if fingers_extended['index'] and fingers_extended['middle'] and not (fingers_extended['ring'] or fingers_extended['pinky']):
            return "Peace", 0.98
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and not fingers_extended['pinky'] and not thumb_extended:
            return "Water", 0.98
        if thumb_extended and fingers_extended['index'] and not (fingers_extended['middle'] or fingers_extended['ring'] or fingers_extended['pinky']):
            return "Medicine", 0.98
        if thumb_extended and fingers_extended['index'] and fingers_extended['pinky'] and not (fingers_extended['middle'] or fingers_extended['ring']):
            return "I Love You", 0.98
        if thumb_extended and fingers_extended['pinky'] and not (fingers_extended['index'] or fingers_extended['middle'] or fingers_extended['ring']):
            return "Thanks", 0.98
        if fingers_extended['index'] and fingers_extended['pinky'] and not (thumb_extended or fingers_extended['middle'] or fingers_extended['ring']):
            return "Where are you going?", 0.98
        if fingers_extended['pinky'] and not (thumb_extended or fingers_extended['index'] or fingers_extended['middle'] or fingers_extended['ring']):
            return "Toilet", 0.98
        if fingers_extended['ring'] and fingers_extended['pinky'] and not (thumb_extended or fingers_extended['index'] or fingers_extended['middle']):
            return "Sleep", 0.98
        if thumb_extended and fingers_extended['ring'] and fingers_extended['pinky'] and not (fingers_extended['index'] or fingers_extended['middle']):
            return "Happy", 0.98

        # --- Single Finger & Thumb Expressions ---
        if fingers_extended['index'] and not (thumb_extended or fingers_extended['middle'] or fingers_extended['ring'] or fingers_extended['pinky']):
            return "No", 0.95
        if thumb_extended and not (fingers_extended['index'] or fingers_extended['middle'] or fingers_extended['ring'] or fingers_extended['pinky']):
            t_tip = pts[4]
            thumb_mcp = pts[2]
            if t_tip[1] < thumb_mcp[1] - 0.01:
                return "Thumbs Up", 0.98
            elif t_tip[1] > thumb_mcp[1] + 0.01:
                return "Thumbs Down", 0.98

        folded_count = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not fingers_extended[name]])
        if folded_count >= 3 and not thumb_extended:
            return "Yes", 0.92

        return "Unknown", 0.0

    def _get_distance_3d(self, p1, p2):
        """Calculate Euclidean distance between two 3D landmarks (Not reliable across different hands due to relative Z)."""
        if hasattr(p1, 'z') and hasattr(p2, 'z'):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
    def _are_hands_touching(self, p1, p2, scale, threshold=0.15, debug_label=None):
        """Helper to determine if two points (from different hands) are physically touching. Must use 2D distance only!"""
        dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) / scale
        if debug_label:
            print(f"[DEBUG DIST] {debug_label}: {dist:.3f} (Threshold: {threshold})")
        return dist < threshold

    def detect_gesture(self, hand_1, handedness_1="Right", hand_2=None, handedness_2=None, mode="words"):
        if not hand_1:
            return "No Hand", 0.0
            
        hand_landmarks = hand_1
        features = self._extract_features(hand_landmarks)

        if DEBUG_PRINT_METRICS:
            print(f"[DEBUG] Index PIP Angle: {features['index_pip_angle']:.1f} | Palm Z Diff: {features['palm_z_diff']:.4f} | Mode: {mode} | folded: {sum([1 for name in ['middle', 'ring', 'pinky'] if not features['fingers_extended'][name]])} | thumb_ext: {features['thumb_extended']} | th_idx_dst: {features['thumb_index_dist']:.2f}")

        if mode == "numbers":
            return self._detect_numbers(features)
        elif mode == "words":
            return self._detect_words(features)
        elif mode == "alphabets":
            return self._detect_alphabets(hand_1, handedness_1, hand_2, handedness_2, features)
        
        return "Unknown", 0.0

    def _detect_alphabets(self, hand_1, handedness_1, hand_2, handedness_2, f1):
        if not hand_1:
            return "Unknown", 0.0
            
        f2 = self._extract_features(hand_2) if hand_2 else None
        
        # Log Handedness stability (Req #2)
        if f2:
            print(f"[DEBUG HANDEDNESS] Input H1: {handedness_1} | Input H2: {handedness_2}")

        # Dynamically assign dominant hand (Right hand by default)
        dom_f, nondom_f = f1, f2
        pts_dom = hand_1.landmark
        pts_nondom = hand_2.landmark if hand_2 else None
        
        if handedness_1 != "Right" and hand_2 and handedness_2 == "Right":
            dom_f, nondom_f = f2, f1
            pts_dom, pts_nondom = hand_2.landmark, hand_1.landmark

        scale = dom_f['scale']
        T = 2.0 # Touching threshold
        
        # --- SINGLE HAND CHECKS ---
        folded_count_dom = sum([1 for name in ['middle', 'ring', 'pinky'] if not dom_f['fingers_extended'][name]])
        
        single_gest = "Unknown"
        single_conf = 0.0

        # Check C vs L
        # L shape: thumb extended SIDEWAYS (horizontal) and only index pointing up — they form a 90-degree L shape.
        # Key: thumb must be horizontal (dx > dy), NOT vertical (which would be U)
        if single_gest == "Unknown" and dom_f['thumb_extended']:
            if dom_f['fingers_extended']['index'] and not dom_f['fingers_extended']['middle'] and folded_count_dom >= 3:
                if dom_f.get('index_pip_angle', 180) > 145:
                    pts = dom_f['pts']
                    thumb_dx = abs(pts[4][0] - pts[2][0])   # horizontal spread of thumb
                    thumb_dy = pts[2][1] - pts[4][1]         # vertical rise of thumb (positive = up)
                    # For L: thumb must be spread away from index (large distance between tips, forming 90 degrees)
                    d_thumb_index = dom_f.get('d_thumb_index', 1.0)
                    if d_thumb_index > 0.6:  # wide gap for L
                        single_gest, single_conf = "L", 0.98

        # J shape: Hooked index pointing vertically.
        # Must be evaluated before C to prevent a vertical hook with thumb out from triggering C.
        if single_gest == "Unknown" and folded_count_dom == 3:
            pts = dom_f['pts']
            dy_mcp_tip = abs(pts[5][1] - pts[8][1])
            dx_mcp_tip = abs(pts[5][0] - pts[8][0])
            if dy_mcp_tip > dx_mcp_tip:  # index pointing more vertically than horizontally
                index_pip_angle = dom_f.get('index_pip_angle', 0)
                if index_pip_angle < 160:  # must be curled/hooked, not straight like I
                    single_gest, single_conf = "J", 0.95

        # C and O share a similar curved index/middle finger base, but differ by the gap
        if single_gest == "Unknown" and not dom_f['fingers_extended']['index'] and not dom_f['fingers_extended']['middle'] and folded_count_dom >= 2:
            if dom_f.get('index_pip_angle', 180) < 160: # Any curved shape
                pts = dom_f['pts']
                d_thumb_index_tip = dom_f.get('d_thumb_index', 1.0)
                d_thumb_index_pip = self._get_distance_2d(pts[4], pts[6]) / scale
                min_gap = min(d_thumb_index_tip, d_thumb_index_pip)
                
                # If the loop is tightly closed (thumb touches index tip or index knuckle), it's an O
                if min_gap < 0.35:
                    single_gest, single_conf = "O", 0.98
                # If there's a clear gap, it's a C
                else:
                    if dom_f.get('thumb_pinky_dist', 0) > 0.35:
                        single_gest, single_conf = "C", 0.95

        # I shape: Index finger pointing to cheek/eye (ISLRTC Sign 1)
        if single_gest == "Unknown" and dom_f['fingers_extended']['index'] and folded_count_dom == 3 and not dom_f['thumb_extended']:
            if dom_f.get('index_pip_angle', 180) > 150: # Must be straight
                single_gest, single_conf = "I", 0.98

        # U shape: Both thumb AND index point straight UP — they are parallel and vertical.
        # Thumb MUST be extended. This is mutually exclusive with J (which requires thumb NOT extended).
        # Key difference from L: in L thumb is HORIZONTAL (sideways), in U thumb is VERTICAL (upward).
        if single_gest == "Unknown" and dom_f['thumb_extended'] and folded_count_dom == 3:
            pts = dom_f['pts']
            # Thumb direction: from MCP (pts[2]) to tip (pts[4])
            thumb_dy = pts[2][1] - pts[4][1]   # positive = thumb tip is ABOVE MCP (pointing up)
            thumb_dx = abs(pts[4][0] - pts[2][0])  # horizontal spread
            # Index direction: from MCP (pts[5]) to tip (pts[8])
            index_dy = pts[5][1] - pts[8][1]   # positive = index tip is ABOVE MCP (pointing up)
            # For U: both must point up AND they must be parallel (tips relatively close together)
            if thumb_dy > thumb_dx * 0.5 and index_dy > 0.05:
                d_thumb_index = dom_f.get('d_thumb_index', 1.0)
                if 0.15 < d_thumb_index < 0.6:  # not pinched into O, but not spread into L
                    single_gest, single_conf = "U", 0.98

        # V shape: Index and middle extended (peace sign), others folded.
        if single_gest == "Unknown" and dom_f['fingers_extended']['index'] and dom_f['fingers_extended']['middle']:
            if not dom_f['fingers_extended']['ring'] and not dom_f['fingers_extended']['pinky']:
                # Thumb is typically tucked/folded over
                if dom_f.get('thumb_pip_angle', 180) < 165 or not dom_f['thumb_extended']:
                    pts = dom_f['pts']
                    dx = pts[8][0] - pts[12][0]
                    dy = pts[8][1] - pts[12][1]
                    index_mid_gap = (dx*dx + dy*dy)**0.5 / scale
                    # Ensure fingers are spread apart to form the V shape
                    if index_mid_gap > 0.15:
                        single_gest, single_conf = "V", 0.98

        if not nondom_f:
            if single_gest != "Unknown":
                return single_gest, single_conf
            else:
                return "Unknown", 0.0

        # --- TWO HAND CHECKS ---
        dom_fingers = dom_f['fingers_extended']
        nondom_fingers = nondom_f['fingers_extended']
        fold_dom = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not dom_fingers[name]])
        fold_nondom = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not nondom_fingers[name]])

        # T: One index vertical, the other index horizontal resting on top of it.
        # This is placed before K and E to resolve conflicts, as T is highly specific in orientation and touch.
        for d, nd, p_d, p_nd in [(dom_f, nondom_f, pts_dom, pts_nondom), (nondom_f, dom_f, pts_nondom, pts_dom)]:
            d_fingers = d['fingers_extended']
            nd_fingers = nd['fingers_extended']
            f_d_fold = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not d_fingers[name]])
            f_nd_fold = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not nd_fingers[name]])
            
            if d_fingers['index'] and f_d_fold == 3 and nd_fingers['index'] and f_nd_fold == 3:
                # Both thumbs tucked
                if (d.get('thumb_pip_angle', 180) < 165 or not d['thumb_extended']) and \
                   (nd.get('thumb_pip_angle', 180) < 165 or not nd['thumb_extended']):
                    # Check strict orientation: d MUST be distinctly horizontal, nd MUST be distinctly vertical
                    dom_dx = abs(p_d[8].x - p_d[5].x)
                    dom_dy = abs(p_d[8].y - p_d[5].y)
                    ndom_dx = abs(p_nd[8].x - p_nd[5].x)
                    ndom_dy = abs(p_nd[8].y - p_nd[5].y)
                    
                    if dom_dx > dom_dy * 1.5 and ndom_dy > ndom_dx * 1.5:
                        # T requires the dominant index to be straight (not hooked like K)
                        if d.get('index_pip_angle', 180) > 150:
                            # Touch point: nd tip (vertical) touches anywhere on the d index finger (tip, body, or base)
                            if any(self._are_hands_touching(p_nd[8], p_d[i], scale, threshold=T*1.5) for i in [5, 6, 7, 8]):
                                return "T", 0.98

        # Check P and Q symmetrically for both hands
        for d_f, nd_f, p_d, p_nd in [(dom_f, nondom_f, pts_dom, pts_nondom), (nondom_f, dom_f, pts_nondom, pts_dom)]:
            if not d_f or not nd_f:
                continue

            nd_fingers = nd_f['fingers_extended']
            d_fingers = d_f['fingers_extended']
            f_nd_fold = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not nd_fingers[name]])

            # Q: Dominant hand forms a loop/circle (thumb+index) touching the non-dominant index tip, loop on the LEFT side (like lowercase 'q')
            if nd_fingers['index'] and f_nd_fold == 3 and d_f['thumb_extended']:
                if not d_fingers['middle'] and not d_fingers['ring'] and not d_fingers['pinky']:
                    # The dominant index finger MUST be curved/bent to form the loop 'O' shape for Q
                    if d_f.get('index_pip_angle', 180) < 160:
                        if self._are_hands_touching(p_d[8], p_nd[8], scale, threshold=T*0.6, debug_label="Q-Index-Tips"):
                            d_thumb_to_dip = np.sqrt((p_d[4].x - p_nd[7].x)**2 + (p_d[4].y - p_nd[7].y)**2) / scale
                            d_thumb_to_mcp = np.sqrt((p_d[4].x - p_nd[5].x)**2 + (p_d[4].y - p_nd[5].y)**2) / scale
                            if d_thumb_to_dip < d_thumb_to_mcp and self._are_hands_touching(p_d[4], p_nd[7], scale, threshold=T*0.8, debug_label="Q-Thumb-to-DIP"):
                                if p_d[5].x < p_nd[5].x:  # dominant hand (loop) is to the LEFT of non-dominant hand
                                    return "Q", 0.98

            # P: Dominant C-shape (thumb + index), thumb tip physically touches PIP (pt[6]) of non-dominant index.
            # This is STRICT — thumb must actually reach and touch the non-dominant PIP joint closely.
            if nd_fingers['index'] and d_f['thumb_extended'] and not d_fingers['middle'] and not d_fingers['ring'] and not d_fingers['pinky']:
                # The dominant index finger MUST be curved/bent to form the 'C' shape for P
                if d_f.get('index_pip_angle', 180) < 175:
                    # In P, the dominant index tip MUST touch the non-dominant index TIP.
                    if self._are_hands_touching(p_d[8], p_nd[8], scale, threshold=T*0.6, debug_label="P-Index-Tips"):
                        d_thumb_to_pip = np.sqrt((p_d[4].x - p_nd[6].x)**2 + (p_d[4].y - p_nd[6].y)**2) / scale
                        d_thumb_to_mcp = np.sqrt((p_d[4].x - p_nd[5].x)**2 + (p_d[4].y - p_nd[5].y)**2) / scale
                        # Thumb must be closer to PIP than to MCP, AND must be truly touching PIP (tight threshold)
                        if d_thumb_to_pip <= d_thumb_to_mcp and self._are_hands_touching(p_d[4], p_nd[6], scale, threshold=T*1.2, debug_label="P-Thumb-to-PIP"):
                            return "P", 0.98

        # D: Right hand 'C' touches straight left index
        if nondom_fingers['index'] and dom_f['thumb_extended'] and not dom_fingers['middle'] and not dom_fingers['ring'] and not dom_fingers['pinky']:
            if self._are_hands_touching(pts_dom[8], pts_nondom[8], scale, threshold=T*0.6, debug_label="D-Index-Tips"):
                if self._are_hands_touching(pts_dom[4], pts_nondom[5], scale, threshold=T*1.2, debug_label="D-Thumb-to-Base") or self._are_hands_touching(pts_dom[4], pts_nondom[6], scale, threshold=T*1.2, debug_label="D-Thumb-to-Base"):
                    return "D", 0.98

        # K (NEW): Non-dominant index points vertically straight up. Dominant index is bent/hooked and touches it.
        # Dominant middle, ring, pinky are folded. Thumb is not used in the sign (so we ignore its state to prevent MediaPipe occlusion errors).
        if nondom_fingers['index'] and fold_nondom == 3 and not dom_fingers['middle'] and not dom_fingers['ring'] and not dom_fingers['pinky']:
            # 1. Non-dominant index MUST be vertical
            nondom_index_dy = abs(pts_nondom[8].y - pts_nondom[5].y)
            nondom_index_dx = abs(pts_nondom[8].x - pts_nondom[5].x)
            if nondom_index_dy > nondom_index_dx * 0.8:
                
                # 2. Dominant index MUST be bent/hooked (not perfectly straight)
                dom_index_pip_angle = dom_f.get('index_pip_angle', 180)
                if dom_index_pip_angle < 160:
                    
                    # 3. Touch check: Dominant index (PIP or Tip) touches the Non-dominant index (middle or tip)
                    if (self._are_hands_touching(pts_dom[6], pts_nondom[6], scale, threshold=T*2.0) or
                        self._are_hands_touching(pts_dom[6], pts_nondom[7], scale, threshold=T*2.0) or
                        self._are_hands_touching(pts_dom[8], pts_nondom[6], scale, threshold=T*2.0) or
                        self._are_hands_touching(pts_dom[8], pts_nondom[7], scale, threshold=T*2.0) or
                        self._are_hands_touching(pts_dom[7], pts_nondom[6], scale, threshold=T*2.0)):
                        return "K", 0.98

        # F: Index and middle crossed (Indian Sign Language "F")
        if dom_fingers['index'] and dom_fingers['middle'] and nondom_fingers['index'] and nondom_fingers['middle']:
            # Ensure ring and pinky are folded to match the exact sign
            if not dom_fingers['ring'] and not dom_fingers['pinky'] and not nondom_fingers['ring'] and not nondom_fingers['pinky']:
                # Check for physical intersection anywhere along the index/middle fingers
                if (self._are_hands_touching(pts_dom[6], pts_nondom[6], scale, threshold=T) or 
                    self._are_hands_touching(pts_dom[8], pts_nondom[8], scale, threshold=T) or
                    self._are_hands_touching(pts_dom[8], pts_nondom[6], scale, threshold=T) or
                    self._are_hands_touching(pts_dom[6], pts_nondom[8], scale, threshold=T)):
                    return "F", 0.98

        # G: Stacked fists
        if fold_dom == 4 and fold_nondom == 4:
            y_diff = abs(pts_dom[0].y - pts_nondom[0].y)
            if y_diff > 0.12: # Stacked vertically
                # Allow either hand on top
                if (self._are_hands_touching(pts_dom[17], pts_nondom[5], scale, threshold=T*1.5) or 
                    self._are_hands_touching(pts_dom[5], pts_nondom[17], scale, threshold=T*1.5) or 
                    self._are_hands_touching(pts_dom[0], pts_nondom[0], scale, threshold=T*1.5)):
                    return "G", 0.98

        # B: Hands bent forming a tent/roof (visible space between palms)
        if self._are_hands_touching(pts_dom[8], pts_nondom[8], scale, threshold=T) and self._are_hands_touching(pts_dom[12], pts_nondom[12], scale, threshold=T):
            # Ensure they form a tent (wrists are far apart)
            if not self._are_hands_touching(pts_dom[0], pts_nondom[0], scale, threshold=T*1.5):
                return "B", 0.98

        # A: Both hands extend thumb only
        if dom_f['thumb_extended'] and nondom_f['thumb_extended']:
            if fold_dom == 4 and fold_nondom == 4:
                return "A", 0.98



        # M: Three fingers flat on palm (Checked before H to prevent collision)
        if dom_fingers['index'] and dom_fingers['middle'] and dom_fingers['ring'] and not dom_fingers['pinky']:
            if nondom_fingers['index'] and nondom_fingers['middle'] and nondom_fingers['ring'] and nondom_fingers['pinky']:
                if self._are_hands_touching(pts_dom[12], pts_nondom[9], scale, threshold=T) or self._are_hands_touching(pts_dom[12], pts_nondom[0], scale, threshold=T):
                    return "M", 0.98

        # N: Two fingers (index, middle) flat on palm
        if dom_fingers['index'] and dom_fingers['middle'] and not dom_fingers['ring'] and not dom_fingers['pinky']:
            if nondom_fingers['index'] and nondom_fingers['middle'] and nondom_fingers['ring'] and nondom_fingers['pinky']:
                if self._are_hands_touching(pts_dom[12], pts_nondom[9], scale, threshold=T) or self._are_hands_touching(pts_dom[12], pts_nondom[0], scale, threshold=T) or self._are_hands_touching(pts_dom[8], pts_nondom[9], scale, threshold=T):
                    return "N", 0.98

        # H: Flat hand over flat hand
        if fold_dom <= 1 and fold_nondom <= 1:
            if (self._are_hands_touching(pts_dom[0], pts_nondom[9], scale, threshold=T) or 
                self._are_hands_touching(pts_dom[9], pts_nondom[9], scale, threshold=T) or
                self._are_hands_touching(pts_dom[9], pts_nondom[0], scale, threshold=T) or
                self._are_hands_touching(pts_dom[12], pts_nondom[9], scale, threshold=T)):
                return "H", 0.98

        # E: Both hands extend index only
        if dom_fingers['index'] and not dom_fingers['middle'] and not dom_fingers['ring'] and not dom_fingers['pinky']:
            if nondom_fingers['index'] and not nondom_fingers['middle'] and not nondom_fingers['ring'] and not nondom_fingers['pinky']:
                # Both thumbs tucked (using angle check for 2D robustness)
                if (dom_f.get('thumb_pip_angle', 180) < 165 or not dom_f['thumb_extended']) and (nondom_f.get('thumb_pip_angle', 180) < 165 or not nondom_f['thumb_extended']):
                    return "E", 0.98

        # M: Three fingers flat on palm
        if dom_fingers['index'] and dom_fingers['middle'] and dom_fingers['ring'] and not dom_fingers['pinky']:
            if self._are_hands_touching(pts_dom[12], pts_nondom[9], scale, threshold=T) or self._are_hands_touching(pts_dom[12], pts_nondom[0], scale, threshold=T):
                return "M", 0.98

        # R: Non-dominant hand flat open, Dominant hand forms 'C' (index and thumb) resting on non-dominant palm
        if fold_nondom == 0:
            if not dom_fingers['middle'] and not dom_fingers['ring'] and not dom_fingers['pinky']:
                if dom_f.get('d_thumb_index', 1.0) > 0.35:  # Thumb and index are apart, forming a C shape
                    touch_idx = any(self._are_hands_touching(pts_dom[8], pts_nondom[i], scale, threshold=T) for i in [0, 1, 2, 5, 9])
                    touch_th = any(self._are_hands_touching(pts_dom[4], pts_nondom[i], scale, threshold=T*1.2) for i in [0, 1, 2, 5, 9, 13, 17])
                    if touch_idx and touch_th:
                        return "R", 0.98

        # S: Both hands have ONLY pinky fingers extended (or hooked). 
        # Hooked fingers fail the 2D extension heuristic, so we don't strictly check fingers_extended['pinky'].
        if not dom_fingers['index'] and not dom_fingers['middle'] and not dom_fingers['ring']:
            if not nondom_fingers['index'] and not nondom_fingers['middle'] and not nondom_fingers['ring']:
                # Thumbs should be tucked/closed
                if (dom_f.get('thumb_pip_angle', 180) < 165 or not dom_f['thumb_extended']) and \
                   (nondom_f.get('thumb_pip_angle', 180) < 165 or not nondom_f['thumb_extended']):
                    # Check if pinky outer joints (PIP, DIP, TIP) are touching (hooked)
                    if any(self._are_hands_touching(pts_dom[i], pts_nondom[j], scale, threshold=T*1.5) for i in [18, 19, 20] for j in [18, 19, 20]):
                        # Ensure fists are somewhat separated (index knuckles not touching, preventing 'G' conflict)
                        if not self._are_hands_touching(pts_dom[5], pts_nondom[5], scale, threshold=T*1.2):
                            return "S", 0.98

        # Hard floor: If we reach the end of the two-handed chain without a match, return Unknown.
        # This prevents single-hand fallbacks from acting as default catchers for invalid two-handed gestures.
        return "Unknown", 0.0
