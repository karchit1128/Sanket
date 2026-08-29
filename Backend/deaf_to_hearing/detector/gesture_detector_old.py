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

    def detect_gesture(self, hand_landmarks, handedness="Right", mode="words"):
        """
        Detects hand gesture accurately based on 21 hand landmarks.
        Returns a tuple of (gesture_name, confidence_value).
        """
        if not hand_landmarks:
            return "No Hand", 0.0

        # Convert landmarks to simple list of [x, y] coordinates in 2D and extract Z for depth
        pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
        pts_z = [getattr(lm, 'z', 0.0) for lm in hand_landmarks.landmark]
        wrist = pts[0]
        
        # Scale reference: Palm size (knuckle width between Index MCP 5 and Pinky MCP 17)
        palm_width = self._get_distance_2d(pts[5], pts[17])
        palm_height = self._get_distance_2d(pts[0], pts[9])
        scale = max(palm_width, palm_height * 0.55)
        if scale < 1e-4:
            scale = 1e-4

        # Anatomical finger joints:
        # Index:  MCP=5,  PIP=6,  DIP=7,  TIP=8
        # Middle: MCP=9,  PIP=10, DIP=11, TIP=12
        # Ring:   MCP=13, PIP=14, DIP=15, TIP=16
        # Pinky:  MCP=17, PIP=18, DIP=19, TIP=20
        # Thumb:  CMC=1,  MCP=2,  IP=3,   TIP=4

        # 1. Determine Extension State of the 4 Main Fingers (Index, Middle, Ring, Pinky)
        # Finger is extended if TIP is further from Wrist than PIP joint AND the joint angle at PIP is straight (> 130 deg)
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
            
            # Primary extended condition: TIP is noticeably further from Wrist than PIP
            is_extended = d_tip > d_pip * 1.05 and d_tip > d_mcp * 1.15
            fingers_extended[name] = is_extended

        # 2. Determine Thumb Extended State
        # Distance from Thumb TIP (4) to Index MCP (5) and Pinky MCP (17) normalized by scale
        t_tip = pts[4]
        i_mcp = pts[5]
        p_mcp = pts[17]
        
        thumb_index_dist = self._get_distance_2d(t_tip, i_mcp) / scale
        thumb_pinky_dist = self._get_distance_2d(t_tip, p_mcp) / scale
        
        # Extended if thumb tip is moved outward away from index knuckle and pinky base
        thumb_extended = thumb_index_dist > 0.38 and thumb_pinky_dist > 0.75

        # 3. Fingertip coordinates
        i_tip = pts[8]
        m_tip = pts[12]
        r_tip = pts[16]
        p_tip = pts[20]

        # Calculate pinch distances between Thumb tip and each finger tip (normalized by scale)
        d_thumb_index = self._get_distance_2d(t_tip, i_tip) / scale
        d_thumb_middle = self._get_distance_2d(t_tip, m_tip) / scale
        d_thumb_ring = self._get_distance_2d(t_tip, r_tip) / scale
        d_thumb_pinky = self._get_distance_2d(t_tip, p_tip) / scale

        # Average pinch distance across all 4 fingers
        avg_all_pinched = (d_thumb_index + d_thumb_middle + d_thumb_ring + d_thumb_pinky) / 4.0

        # Spread metrics
        tip_spread = (self._get_distance_2d(i_tip, m_tip) + 
                      self._get_distance_2d(m_tip, r_tip) + 
                      self._get_distance_2d(r_tip, p_tip)) / scale

        # Advanced Metrics for Numbers
        index_pip_angle = self._get_angle(pts[5], pts[6], pts[8])
        palm_z_diff = abs(pts_z[5] - pts_z[17])
        thumb_pip_angle = self._get_angle(pts[1], pts[2], pts[4])

        if DEBUG_PRINT_METRICS:
            print(f"[DEBUG] Index PIP Angle: {index_pip_angle:.1f} | Palm Z Diff: {palm_z_diff:.4f} | Mode: {mode}")

        # =========================================================================
        # NUMBER RECOGNITION CLASSIFICATION (Mode: "numbers")
        # =========================================================================
        if mode == "numbers":
            folded_count = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not fingers_extended[name]])
            
            # 0: Fist / 'O'
            # (Either completely folded fist, OR the thumb is touching the folded fingertips forming an 'O')
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
                # Enforce direction: Thumb must be pointing UP (tip Y < base Y)
                if pts[4][1] < pts[2][1] - 0.05:
                    # Disambiguate from Thumbs Up using Z-depth tolerance
                    if palm_z_diff < TEST_PALM_Z_TOLERANCE:
                        return "6", 0.95
                    
            # 7: Index and Thumb hooked/curved (C-shape), others folded
            if folded_count >= 3 and not fingers_extended['middle'] and not fingers_extended['ring'] and not fingers_extended['pinky']:
                # The index and thumb are neither fully extended nor fully folded tight.
                # We check the PIP angle for the curve.
                if TEST_HOOK_ANGLE_MIN <= index_pip_angle <= TEST_HOOK_ANGLE_MAX:
                    return "7", 0.95
                    
            # 8: Thumb, Index, Middle extended, others folded
            if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and not fingers_extended['ring'] and not fingers_extended['pinky']:
                return "8", 0.98
                
            # 9: Middle, Ring, Pinky extended, Index on Thumb (pinched)
            if fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']:
                # Index on thumb (pinched)
                if d_thumb_index < 0.35:
                    return "9", 0.98
            
            return "Unknown", 0.0

        # =========================================================================
        # WORD RECOGNITION CLASSIFICATION (Mode: "words")
        # =========================================================================

        # --- Pinched & Touch Gestures ---
        # 1. Food / Eat (All fingertips gathered together into a cone pinch)
        if avg_all_pinched < 0.48 and d_thumb_pinky < 0.50:
            return "Food", 0.98

        # 2. OK Gesture (Index and Thumb tips touching, Middle, Ring, Pinky extended)
        if d_thumb_index < 0.38 and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']:
            return "OK", 0.98

        # 3. Money / Cost (Thumb and Middle tips pinched; Index, Ring, Pinky extended)
        if d_thumb_middle < 0.35 and fingers_extended['index'] and fingers_extended['ring'] and fingers_extended['pinky']:
            return "Money", 0.98

        # 4. Attention (Thumb and Pinky tips pinched; Index, Middle, Ring extended)
        if d_thumb_pinky < 0.30 and fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring']:
            return "Attention", 0.98

        # --- Special Multi-Finger Combo Shapes ---
        # 5. Read / Book (Index, Middle, Pinky extended; Ring & Thumb folded)
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['pinky'] and not fingers_extended['ring']:
            return "Read", 0.98

        # 6. Emergency (Index folded; Thumb, Middle, Ring, Pinky extended)
        if thumb_extended and not fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']:
            return "Emergency", 0.98

        # 7. Question (Ring folded; Thumb, Index, Middle, Pinky extended)
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and not fingers_extended['ring'] and fingers_extended['pinky']:
            return "Question", 0.98

        # 8. Good Morning (Pinky folded; Thumb, Index, Middle, Ring extended)
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and not fingers_extended['pinky']:
            return "Good Morning", 0.98

        # --- Flat Open Hand Expressions: Hello, Stop, Please ---
        all_four_extended = fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and fingers_extended['pinky']
        if all_four_extended:
            # If thumb is tucked across palm towards pinky base
            if thumb_pinky_dist <= 0.78:
                return "Please", 0.98
            else:
                # Distinguish spread hand (Hello) vs closed palm (Stop)
                tip_width = self._get_distance_2d(i_tip, p_tip) / scale
                if tip_width > 1.25 or tip_spread > 0.85:
                    return "Hello", 0.95
                else:
                    return "Stop", 0.95

        # --- 3-Finger and 2-Finger Shapes ---
        # 9. How are you? (Thumb, Index, Middle extended; Ring & Pinky folded)
        if thumb_extended and fingers_extended['index'] and fingers_extended['middle'] and not (fingers_extended['ring'] or fingers_extended['pinky']):
            return "How are you", 0.98

        # 10. Peace (V-Sign: Index and Middle extended; Ring, Pinky, and Thumb folded)
        if fingers_extended['index'] and fingers_extended['middle'] and not (fingers_extended['ring'] or fingers_extended['pinky']):
            return "Peace", 0.98

        # 11. Water (W-Sign: Index, Middle, Ring extended; Thumb and Pinky folded)
        if fingers_extended['index'] and fingers_extended['middle'] and fingers_extended['ring'] and not fingers_extended['pinky'] and not thumb_extended:
            return "Water", 0.98

        # 12. Medicine (L-Shape: Thumb and Index extended; Middle, Ring, Pinky folded)
        if thumb_extended and fingers_extended['index'] and not (fingers_extended['middle'] or fingers_extended['ring'] or fingers_extended['pinky']):
            return "Medicine", 0.98

        # 13. I Love You (Thumb, Index, Pinky extended; Middle & Ring folded)
        if thumb_extended and fingers_extended['index'] and fingers_extended['pinky'] and not (fingers_extended['middle'] or fingers_extended['ring']):
            return "I Love You", 0.98

        # 14. Thanks / Shaka (Thumb & Pinky extended; Index, Middle, Ring folded)
        if thumb_extended and fingers_extended['pinky'] and not (fingers_extended['index'] or fingers_extended['middle'] or fingers_extended['ring']):
            return "Thanks", 0.98

        # 15. Where are you going? (Horns shape: Index & Pinky extended; Thumb, Middle, Ring folded)
        if fingers_extended['index'] and fingers_extended['pinky'] and not (thumb_extended or fingers_extended['middle'] or fingers_extended['ring']):
            return "Where are you going?", 0.98

        # 16. Toilet (Pinky extended only; all others folded)
        if fingers_extended['pinky'] and not (thumb_extended or fingers_extended['index'] or fingers_extended['middle'] or fingers_extended['ring']):
            return "Toilet", 0.98

        # 17. Sleep (Ring & Pinky extended; Thumb, Index, Middle folded)
        if fingers_extended['ring'] and fingers_extended['pinky'] and not (thumb_extended or fingers_extended['index'] or fingers_extended['middle']):
            return "Sleep", 0.98

        # 18. Happy (Thumb, Ring, Pinky extended; Index & Middle folded)
        if thumb_extended and fingers_extended['ring'] and fingers_extended['pinky'] and not (fingers_extended['index'] or fingers_extended['middle']):
            return "Happy", 0.98

        # --- Single Finger & Thumb Expressions ---
        # 19. No (Pointing index finger up; all others folded)
        if fingers_extended['index'] and not (thumb_extended or fingers_extended['middle'] or fingers_extended['ring'] or fingers_extended['pinky']):
            return "No", 0.95

        # 20. Thumbs Up / Thumbs Down (Only thumb extended, all fingers folded)
        if thumb_extended and not (fingers_extended['index'] or fingers_extended['middle'] or fingers_extended['ring'] or fingers_extended['pinky']):
            thumb_mcp = pts[2]
            if t_tip[1] < thumb_mcp[1] - 0.01:
                return "Thumbs Up", 0.98
            elif t_tip[1] > thumb_mcp[1] + 0.01:
                return "Thumbs Down", 0.98

        # 21. Yes (Fist: All 4 fingers folded, thumb folded)
        folded_count = sum([1 for name in ['index', 'middle', 'ring', 'pinky'] if not fingers_extended[name]])
        if folded_count >= 3 and not thumb_extended:
            return "Yes", 0.92

        return "Unknown", 0.0

