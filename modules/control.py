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

def rotate_away(ad):
    '''Rotates the penguin away from the camera.

    Args:
        ad (tuple): A tuple containing:
            angle (float): The current angle of the penguin.
            direction (bool): The current direction the penguin is facing (True if towards the camera, False if away).
    Returns:
        direction (bool): The new direction the penguin is facing after rotation.
    '''
    angle, direction = ad
    match direction or abs(angle-90) > 2:
        case True:  # facing toward the camera
            pyautogui.keyDown('right')
        case False:  # facing away from the camera
            pyautogui.keyUp('right')
    time.sleep(0.002)

    return angle, direction # return the current direction

def rotate_toward_splash(point, visual, tolerance=5, max_time=3.0):
    """Rotate penguin toward splash point, shortest path, handling facing direction."""
    start_time = time.time()
    screen_width, screen_height = vision.get_screen_size()
    splash_x, splash_y = point

    # Compute splash angle (0°=right, 90°=up)
    dx = splash_x - screen_width / 2
    dy = screen_height / 2 - splash_y
    splash_angle = ((math.degrees(math.atan2(dy, dx)) + 270) % 360)/2
    print(f"Splash at ({splash_x}, {splash_y}) → {splash_angle:.1f}°")

    while time.time() - start_time < max_time:
        angle, direction = visual.update_player_detector()

        # Adjust if penguin facing camera (reverse)
        if direction:  
            angle = (angle + 180) % 360

        # Compute shortest angular difference [-180°, 180°]
        diff = (splash_angle - angle + 180) % 360 - 180

        if abs(diff) <= tolerance:
            break  # within tolerance

        # Choose rotation direction
        if diff > 0:
            pyautogui.keyDown('right')
            pyautogui.keyUp('left')
        else:
            pyautogui.keyDown('left')
            pyautogui.keyUp('right')
        time.sleep(0.02)

    pyautogui.keyUp('right')
    pyautogui.keyUp('left')
    print(f"Final angle: {angle:.1f}°, facing {'forward' if not direction else 'toward camera'}, diff={diff:.1f}°")

def searching(visual,debug=False):
    '''Rotates the camera (mouse movement) to search for splashes.
    Args:
        visual (vision.VisionCortex): The vision class instance for updating player detector.
    '''
    # Get angle and direction
    point = visual.update_splash_detector()
    if point:
        if debug: print(f"Splash found at {point}")
        return point  # splash found, exit searching
    else:
        angle, direction = visual.update_player_detector()
        movelength = 100
        rotationspeed = 0.3

        # Rotate the camera (mouse) left and right
        if direction:
            pyautogui.moveRel(-movelength, 0, duration=rotationspeed)  # move mouse left
        else:
            pyautogui.moveRel(movelength, 0, duration=rotationspeed)   # move mouse right

def cast_rod(point,debug=False):
    '''Takes the XY coordinates of the splash point, calculates the distance from the center, and casts for a duration dependent on that distance.'''
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