import numpy as np
from detector.gesture_detector import GestureDetector as NewGestureDetector
from detector.gesture_detector_old import GestureDetector as OldGestureDetector
import sys

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(np.random.random(), np.random.random(), np.random.random()) for _ in range(21)]

def run_test():
    new_detector = NewGestureDetector()
    old_detector = OldGestureDetector()
    
    print("Running Regression Test for 100,000 random hand poses...")
    np.random.seed(42)
    mismatches = 0
    for i in range(100000):
        hm = MockHandLandmarks()
        for mode in ["numbers", "words"]:
            old_res, old_conf = old_detector.detect_gesture(hm, mode=mode)
            new_res, new_conf = new_detector.detect_gesture(hm, mode=mode)
            
            if old_res != new_res or abs(old_conf - new_conf) > 1e-5:
                print(f"Mismatch at iteration {i} for mode {mode}!")
                print(f"Old: {old_res}, {old_conf}")
                print(f"New: {new_res}, {new_conf}")
                mismatches += 1
                if mismatches > 5:
                    return False

    print("Success! 100% match between old and new logic.")
    return mismatches == 0

if __name__ == "__main__":
    if not run_test():
        sys.exit(1)
    sys.exit(0)
