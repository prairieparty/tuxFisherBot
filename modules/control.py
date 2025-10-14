import pyautogui
import time

def enterWindow():
    '''Clicks the center of the screen to ensure the game window is active.'''
    screen_width, screen_height = pyautogui.size()
    pyautogui.click(screen_width // 2, screen_height // 2)
    # drag the mouse down a little bit from the center point so the penguin is looking dead on
    # pyautogui.move(0, 50, duration=0.2)