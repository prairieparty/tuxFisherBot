#!/usr/bin/env python3

# Author: Jackson O'Connor
# Date: 20251009
# Description: This program automatically launches Tux Fisher, before proceeding to automatically catch fish.
'''Patch Notes:
20251001: Initial commit.
- Added basic structure.
- Added imports.
- Added image & module directory search.
20251002:
- Added function to launch Tux Fisher.
- Introduced debug module with live overlay (not to be included in final build).
- Tried various image recognition methods (template matching, pyautogui, ORB).
- Added function to debug ORB keypoints.
20251008:
- Set up GitHub repository.
- Updated ORB parameters in vision module to match debug model.
- Condensed variables in main.py to their respective sections.
- Began developing detection for player avatar angle (vision module).
- Moved player images to images/player directory & updated code accordingly.
20251009:
- Started debugging player angle detection with matched keypoint geometry (debug module).
- Added black & white masking functions to debug module to assist with player angle detection.
- Changed screenshot method in vision module to use mss for better performance + ROI selection.
20251014:
- Fixed bug in screen capture using Jean's code from ImageCap.py.
- added new module loading function to condense code.
- made a working angle detection class - uses the white mask to find the penguin's general direction, and fishing rod to determine tilt angle.
20251017:
- Built a VisionCortex class to encapsulate vision functions.
- Corrected splash ROI to be dynamic based on screen size.
- Began function to rotate avatar toward splash point (control module).
- Made player rotate until facing away from camera at start of main loop.
'''
'''To Do:

- correct splash function to actually work reliably *or at all* (vision module)

- create function to rotate avatar to face fish (control module)

- determine distance to fish (vision module)

- create function to cast line (control module)

- create "search" mode to look around the environment for fish (if it doesn't introduce false positives into ORB detection)

'''

# IMPORTS

import cv2 as cv
from mss import mss
import numpy as np
import pyautogui
import time
import os
from pathlib import Path
import sys
from importlib.util import spec_from_file_location, module_from_spec

# VARIABLES

# Find images & modules directory
home_path = Path("/home/")
matching_files = home_path.rglob("*tuxfishing.conf") # Recursively searches for the tuxfishing.conf file within the images directory.
tuxconfs = list(matching_files) # Converts the generator to a list
# determine which tuxfishing.conf is within the images directory, and which is within modules directory
if len(tuxconfs) == 0:
    raise FileNotFoundError("No tuxfishing.conf file found in /home/ directory or its subdirectories.")
elif len(tuxconfs) == 1:
    raise FileNotFoundError("Only one tuxfishing.conf file found. Please ensure there are two: one in the images directory and one in the modules directory.")
else:
    #Determine which is which
    if "images" in str(tuxconfs[0].parent):
        images_directory = tuxconfs[0].parent
        modules_directory = tuxconfs[1].parent
    elif "images" in str(tuxconfs[1].parent):
        images_directory = tuxconfs[1].parent
        modules_directory = tuxconfs[0].parent
    else:
        raise FileNotFoundError("Could not determine which tuxfishing.conf file is in the images directory. Please ensure one is in the images directory and one is in the modules directory.")

# Find all image files in the images directory
image_files = list(Path(images_directory).rglob("*.png")) + list(Path(images_directory).rglob("*.jpg")) # Add more image formats as needed
# Find fullscreen button image
fullscreen_button = [img for img in image_files if img.name == "fullscreen_icon.png"][0]
# Find all images in the player subfolder
player_images = list(Path(images_directory / "player").rglob("*.png")) + list(Path(images_directory / "player").rglob("*.jpg")) # Add more image formats as needed
# Reorder player images to start with back & go clockwise
player_images_ordered = []
try:
    assert len(player_images) == 8, "Expected 8 player images (front, back, left, right, NE, NW, SE, SW). Please ensure all are present in the player images directory."
except AssertionError as e:
    print(e)
for name in ["player_front.png", "player_SE.png", "player_right.png", "player_NE.png", "player_back.png", "player_NW.png", "player_left.png", "player_SW.png"]:
    player_images_ordered.append([img for img in player_images if img.name == name][0])
player_images = player_images_ordered



# FUNCTIONS

def loadModule(module_name):
    module_spec = spec_from_file_location(module_name, str([mf for mf in module_files if mf.name == f"{module_name}.py"][0]))
    module = module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module

def launch_tux_fisher(delay=5, fullscreen=True):
    # Open Tux Fishing
    os.popen("xdg-open https://pushergames.itch.io/tuxfishing") # opens the web version of Tux Fishing
    time.sleep(delay) # Wait for Tux Fishing to open
    # fullscreen the game by finding the fullscreen button and clicking it
    if fullscreen:
        if fullscreen_button:
            try:
                fullscreen_location = pyautogui.locateCenterOnScreen(str(fullscreen_button), confidence=0.8)
                if fullscreen_location:
                    pyautogui.click(fullscreen_location)
                    time.sleep(3) # Wait for the game to go fullscreen
                else:
                    print("Fullscreen button not found on screen.")
            except Exception as e:
                print(f"Error locating fullscreen button: {e}")
    else:
        print("Fullscreen button image not found in images directory.")

# Main Logic
def main():
    # Find and load modules
    global module_files
    module_files = list(Path(modules_directory).rglob("*.py"))

    # make modules accessible globally
    global control, vision, debug

    # load the vision module
    vision = loadModule("vision")
    eyes = vision.VisionCortex(debug=True) # initialize vision class with debug mode on
    # load the control module
    control = loadModule("control")
    # load the debug module (not to be included in final build)
    debug = loadModule("debug")

    # Launch Tux Fisher
    launch_tux_fisher(fullscreen=True)
    
    control.enterWindow() # Ensure the game window is active

    direction = True # assume starting facing toward the camera
    angle = 0.0

    while direction or angle < 100: # Rotate away from camera until facing away
        ad = eyes.update_player_detector()
        angle, direction = control.rotate_away(ad)

    # run for 30 seconds and locate splashes
    start_time = time.time()
    while time.time() - start_time < 30:

        # find player angle
        angle, direction = eyes.update_player_detector()

        # Locate splashes using ORB
        point = eyes.update_splash_detector()
        
        # If a splash is found, rotate toward it
        if point:
            splash_x, splash_y = point
            control.rotate_toward_splash(splash_x, splash_y, angle, direction)
        
        time.sleep(0.1) # Wait before next search

if __name__ == "__main__":
    main()