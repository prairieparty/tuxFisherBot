import pyautogui
import vision
import math
import time

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
    tolerance=7,
    sample_delay=0.02,
    base_step_time=0.06,
    min_step_time=0.02,
    slowdown_radius=15,
    debug=False
):
    """
    Rotates the penguin until it's facing forward (angle ≈ 90°, direction=False),
    using adaptive slowdown near the target to prevent overshoot.

    Args:
        visual (VisionCortex): Vision system providing update_player_detector().
        target_angle (float): Target angle for facing forward (default=90°).
        tolerance (float): Acceptable deviation (±degrees).
        sample_delay (float): Delay between detector samples.
        base_step_time (float): Maximum keypress duration (fast rotation).
        min_step_time (float): Minimum keypress duration (fine adjustment).
        slowdown_radius (float): Begin slowdown when within this many degrees of target.
    """
    print("[rotate_away] Rotating penguin until it faces forward...")

    while True:
        result = visual.update_player_detector()
        if result is None:
            print("[rotate_away] Lost tracking; pausing...")
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            time.sleep(0.2)
            continue

        angle, facing_forward = result
        angle = angle % 360

        # --- Stop when facing forward (≈90°, not camera-facing) ---
        if abs(angle - target_angle) <= tolerance and not facing_forward:
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            print(f"[rotate_away] Penguin aligned forward at {angle:.2f}° ✅")
            break

        # --- Compute angular difference ---
        diff = ((target_angle - angle + 540) % 360) - 180
        abs_diff = abs(diff)
        if debug:
            print(f"[rotate_away] angle={angle:.2f}°, diff={diff:.2f}°, facing_forward={facing_forward}")

        # --- Adaptive slowdown ---
        if abs_diff < slowdown_radius:
            # Map proximity to step duration (closer = slower)
            # Linear scale: diff=slowdown_radius → base_step_time, diff=0 → min_step_time
            scale = abs_diff / slowdown_radius
            step_time = min_step_time + (base_step_time - min_step_time) * scale
        else:
            step_time = base_step_time

        # --- Turn shortest direction ---
        if diff > 0:
            pyautogui.keyDown('right')
            time.sleep(step_time)
            pyautogui.keyUp('right')
        else:
            pyautogui.keyDown('left')
            time.sleep(step_time)
            pyautogui.keyUp('left')

        time.sleep(sample_delay)

    return angle, facing_forward

def rotate_camera_toward_splash(sx, sy, visual, 
                                x_tolerance=40, 
                                y_tolerance=80, 
                                sensitivity=0.004, 
                                max_time=3.0, 
                                vertical=False):
    """
    Rotate the camera until the splash point (sx, sy) is centered on screen.

    Args:
        sx, sy (float): Splash point in absolute screen coordinates.
        visual (VisionCortex): Vision instance (for screen center, size, etc.).
        x_tolerance (int): Pixel distance in x to stop (horizontal centering).
        y_tolerance (int): Pixel distance in y to stop (vertical centering).
        sensitivity (float): Mouse movement per pixel offset (tune for camera rotation speed).
        max_time (float): Max time allowed to adjust (seconds).
        vertical (bool): Whether to adjust vertically as well.
    """

    cx, cy = visual.screen_center
    start_time = time.time()

    while time.time() - start_time < max_time:
        # calculate offset from screen center
        dx = sx - cx
        dy = sy - cy

        # stop if centered enough
        if abs(dx) <= x_tolerance and (not vertical or abs(dy) <= y_tolerance):
            print(f"Splash centered (dx={dx:.1f}, dy={dy:.1f})")
            break

        # proportional horizontal movement
        move_x = -dx * sensitivity
        move_y = 0

        # optional vertical centering
        if vertical:
            move_y = dy * sensitivity * 0.7

        pyautogui.moveRel(move_x, move_y, duration=0.01)

        # progressively slow down near center
        sensitivity *= 0.95 if abs(dx) < 200 else 1.0

        # small pause to allow frame to update
        time.sleep(0.015)

    # stop movement
    pyautogui.moveRel(0, 0)
    print("Camera alignment complete.")

def searching(visual,debug=False):
    '''Rotates the camera (mouse movement) to search for splashes.
    Args:
        visual (vision.VisionCortex): The vision class instance for updating player detector.
    '''
    # Get angle and direction
    visual.update_player_detector()
    # Look for splashes
    visual.motion_detection() #call it without storing so it doesnt pick up camera movement as splash
    time.sleep(0.2) # small delay to allow frame to update
    points = visual.motion_detection()
    if points is not None and len(points) > 0:
        if debug: [print(f"  Found splash at {i}") for i in points]
        return points[0]  # splash found, exit searching
    else:
        movelength = 100
        rotationspeed = 0.3
        pyautogui.moveRel(movelength, 0, duration=rotationspeed)   # move mouse right
        visual.splashROI = (0, visual.screen_size[1]//8,
                  visual.screen_size[0], visual.screen_size[1]//4)

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