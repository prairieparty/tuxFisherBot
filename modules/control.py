import pyautogui
import vision
import math
import time

def enterWindow():
    '''Clicks the penguin to ensure the game window is active.'''
    screen_width, screen_height = vision.get_center_of_screen()
    # move mouse to center before clicking
    pyautogui.moveTo(screen_width // 2, screen_height // 2)
    time.sleep(0.1)
    pyautogui.click(screen_width // 2, screen_height // 2)
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
    match direction or angle < 100:
        case True:  # facing toward the camera
            pyautogui.keyDown('right')
        case False:  # facing away from the camera
            pyautogui.keyUp('right')
    time.sleep(0.2)

    return angle, direction # return the current direction

def rotate_toward_splash(point, angle, direction, tolerance=5):
    '''Rotates the penguin towards the splash point.
    Args:
        point (tuple): The (x, y) coordinates of the splash point.
        angle (float): The current angle of the penguin.
        direction (bool): The current direction the penguin is facing (True if towards the camera, False if away).
        tolerance (float): The tolerance angle for rotation.
    '''
    # Calculate the angle to the splash point
    splash_angle = math.degrees(math.atan2(point[1] - vision.get_center_of_screen()[1], point[0] - vision.get_center_of_screen()[0]))
    # Rotate until the penguin is facing the splash point
    while abs(splash_angle - angle) > tolerance:
        if (splash_angle - angle + 360) % 360 < 180:
            # Rotate right
            pyautogui.press('right')
        else:
            # Rotate left
            pyautogui.press('left')
        time.sleep(0.1)
        angle, direction = vision.determine_avatar_angle()
    print(f"Rotated to angle: {angle:.1f}° facing {'forward' if direction else 'backward'}.")