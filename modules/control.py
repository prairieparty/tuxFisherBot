import pyautogui
import vision
import math
import time
import numpy as np
import csv

def enterWindow():
    '''Clicks the penguin to ensure the game window is active.'''
    screen_width, screen_height = vision.get_screen_size()
    # move mouse to center before clicking
    pyautogui.moveTo(screen_width // 2, screen_height // 2, duration=0.2)
    time.sleep(0.1)
    pyautogui.mouseDown(); pyautogui.mouseUp()
    # drag the mouse down a little bit from the center point so the penguin is looking dead on
    # pyautogui.move(0, 50, duration=0.2)

def rotate_away(
    visual,
    target_angle=90,
    tolerance=12,
    sample_delay=0.02,
    base_step_time=0.06,
    min_step_time=0.02,
    slowdown_radius=20,
    max_time=6.0,
    debug=False
):
    """
    Rotate until facing forward (angle≈target_angle). Uses raw angle, adaptive step time,
    and one-time self-calibration to map left/right keys to angle increase/decrease.
    """

    def shortest_diff(a, b):
        # signed shortest difference a-b in degrees, in (-180, 180]
        return ((a - b + 540.0) % 360.0) - 180.0

    def read_raw():
        val = visual.update_player_detector(smooth_window=1)
        return None if val is None else (val % 360.0)

    def tap(key, dt):
        pyautogui.keyDown(key)
        time.sleep(dt)
        pyautogui.keyUp(key)

    # ensure keys not stuck
    pyautogui.keyUp('left'); pyautogui.keyUp('right')

    # establish initial reading
    angle = read_raw()
    if angle is None:
        if debug: print("[rotate_away/dbg] No initial angle; waiting for tracking...")
        t_wait = time.time()
        while angle is None and time.time() - t_wait < 1.5:
            time.sleep(0.05)
            angle = read_raw()
        if angle is None:
            if debug: print("[rotate_away/dbg] Tracking failed.")
            return None

    # self-calibrate which key increases angle (store on visual)
    if not hasattr(visual, "_turn_left_sign"):
        a0 = angle
        tap('left', 0.03)
        time.sleep(sample_delay)
        a1 = read_raw() or a0
        left_sign = np.sign(shortest_diff(a1, a0)) or 1  # fallback +1
        visual._turn_left_sign = int(left_sign)
        if debug: print(f"[rotate_away/dbg] Calibrated: left_sign={visual._turn_left_sign}")

    t0 = time.time()
    last_angle = angle
    no_progress = 0

    while time.time() - t0 < max_time:
        angle = read_raw()
        if angle is None:
            if debug: print("[rotate_away/dbg] Lost tracking; pausing...")
            time.sleep(0.1)
            continue

        diff = shortest_diff(target_angle, angle)
        abs_diff = abs(diff)
        if debug:
            print(f"[rotate_away] angle={angle:.2f}°, diff={diff:.2f}°")

        # stop condition
        if abs_diff <= tolerance:
            pyautogui.keyUp('left'); pyautogui.keyUp('right')
            if debug:
                print(f"[rotate_away] Penguin aligned forward at {angle:.2f} degrees within +/-{tolerance} degrees tolerance.")
            return angle

        # adaptive step time
        if abs_diff < slowdown_radius:
            step = min_step_time + (base_step_time - min_step_time) * (abs_diff / max(slowdown_radius, 1.0))
        else:
            step = base_step_time
        step = float(np.clip(step, min_step_time, base_step_time))

        # choose direction based on calibration and sign of desired change
        need_sign = 1 if diff > 0 else -1
        if visual._turn_left_sign == need_sign:
            tap('left', step)
        else:
            tap('right', step)

        time.sleep(sample_delay)

        # progress check
        new_angle = read_raw() or angle
        moved = abs(shortest_diff(new_angle, last_angle))
        if moved < 0.5:
            no_progress += 1
            # mildly increase step to break static friction
            base_step_time = min(0.10, base_step_time + 0.01)
        else:
            no_progress = 0
        last_angle = new_angle

        # re-check mapping if repeatedly stuck
        if no_progress >= 6:
            if debug: print("[rotate_away/dbg] Recalibrating turn direction...")
            a0 = read_raw() or last_angle
            tap('left', 0.04)
            time.sleep(sample_delay)
            a1 = read_raw() or a0
            left_sign = np.sign(shortest_diff(a1, a0)) or visual._turn_left_sign
            visual._turn_left_sign = int(left_sign)
            no_progress = 0

    pyautogui.keyUp('left'); pyautogui.keyUp('right')
    if debug:
        cur = read_raw()
        print(f"[rotate_away] Timeout; last angle={cur if cur is not None else 'None'}")
    return angle

def rotate_camera_toward_splash(sx, sy, visual, 
                                x_tolerance=40, 
                                y_tolerance=80, 
                                sensitivity=0.04, 
                                max_time=3.0, 
                                vertical=False,
                                min_step=3.0,
                                max_step=100.0,
                                reacquire=True,
                                reacquire_period=0.08):
    """
    Center the splash (sx, sy) by rotating the camera with repeated mouse moves.
    Closed-loop: optionally re-acquires the splash each step using visual.motion_detection.
    Returns True on success, False on timeout.
    """
    # Resolve screen center
    sc = getattr(visual, "screen_center", None)
    if not sc or not isinstance(sc, (tuple, list)) or len(sc) < 2:
        sc = vision.get_center_of_screen()
    cx, cy = int(sc[0]), int(sc[1])

    # Track a current target that we can refresh
    cur_x, cur_y = float(sx), float(sy)
    roi_x, roi_y, _, _ = getattr(visual, "splashROI", (0, 0, 0, 0))

    start = time.time()
    last_acq = 0.0
    centered = False
    dx = cur_x - cx
    dy = cur_y - cy

    while time.time() - start < max_time:
        # Reacquire splash to update target (closed loop)
        if reacquire and (time.time() - last_acq) >= reacquire_period:
            pts = visual.motion_detection()
            if pts is not None:
                pts = np.asarray(pts)
                if pts.size >= 2 and pts.ndim == 2 and pts.shape[1] >= 2:
                    # convert ROI-relative points to screen coords
                    xs = pts[:, 0] + roi_x
                    ys = pts[:, 1] + roi_y
                    # pick robust center of detections
                    tx, ty = visual._robust_centroid(xs, ys)
                    cur_x, cur_y = float(tx), float(ty)
            last_acq = time.time()

        # Compute current error to screen center
        dx = cur_x - cx
        dy = cur_y - cy

        # Stop condition
        if abs(dx) <= x_tolerance and (not vertical or abs(dy) <= y_tolerance):
            centered = True
            break

        # Proportional steps (no prediction; we will re-measure next loop)
        step_x = np.sign(dx) * max(min_step, min(abs(dx) * sensitivity, max_step))
        step_y = 0.0
        if vertical:
            step_y = np.sign(dy) * max(1.0, min(abs(dy) * (0.7 * sensitivity), 0.7 * max_step))

        pyautogui.moveRel(step_x, step_y, duration=0.01)
        time.sleep(0.012)

    # stop movement (explicit no-op)
    pyautogui.moveRel(0, 0)
    if centered:
        print(f"[rotate_toward_splash] Camera aligned (dx={dx:.1f}, dy={dy:.1f}).")
        return True
    else:
        print(f"[rotate_toward_splash] Timeout; residual (dx={dx:.1f}, dy={dy:.1f}).")
        return False
    
class Searcher():
    '''Class to encapsulate searching behavior.'''
    def __init__(self, visual, maximal=8, debug=False):
        self.visual = visual
        self.searchCount = 0
        self.searchMax = maximal #how many rotations before going the other way
        self.moveLength = 100
        self.rotationSpeed = 0.3
        self.debug = debug

    def searching(self):
        '''Rotates the camera (mouse movement) to search for splashes.
        Args:
            visual (vision.VisionCortex): The vision class instance for updating player detector.
        '''
        # Get angle and direction
        self.visual.update_player_detector()
        # Look for splashes
        self.visual.motion_detection() #call it without storing so it doesnt pick up camera movement as splash
        time.sleep(0.2) # small delay to allow frame to update
        points = self.visual.motion_detection()
        if points is not None and len(points) > 0:
            if self.debug: [print(f"  Found splash at {i}") for i in points]
            return points[0]  # splash found, exit searching
        else:
            pyautogui.moveRel(self.moveLength, 0, duration=self.rotationSpeed)   # move mouse right
            self.visual.splashROI = (0, self.visual.screen_size[1]//8,
                    self.visual.screen_size[0], self.visual.screen_size[1]//4)
            self.searchCount += 1
            if self.searchCount >= self.searchMax:
                self.moveLength = -self.moveLength  # reverse direction
                self.searchCount = 0
                if self.debug: print("Reversing search direction.")
            return None  # no splash found

def cast_rod(point,debug=False):
    '''Takes the XY coordinates of the splash point, calculates the distance from the center, and casts for a duration dependent on that distance.'''
    # stop all rotation
    pyautogui.keyUp("left")
    pyautogui.keyUp("right")

    #cast
    if point:
        center_x = vision.get_screen_size()[0] // 2
        center_y = vision.get_screen_size()[1] // 2
        distance = math.hypot(point[0] - center_x, point[1] - center_y)
        duration = min(max(distance / 1000, 0.1), 2.0)  # Scale distance to duration
        if debug:
            print(f"Casting rod for distance {distance:.1f} pixels, duration {duration:.2f} seconds.")
        pyautogui.mouseDown()
        time.sleep(duration)
    pyautogui.mouseUp()