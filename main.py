#!/usr/bin/env python3

# Author: Jackson O'Connor
# Date: 20251001
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

'''
'''To Do:
- create segregated control module (pyautogui)
- create "search" mode to look around the environment for fish (if it doesn't introduce false positives into ORB detection)
- implement image recognition to detect fish
- create function to catch fish using control module
'''

# Imports

import cv2 as cv
import numpy as np
import pyautogui
import time
import os
from pathlib import Path
import sys
from importlib.util import spec_from_file_location, module_from_spec

# Variables

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

# Find and load modules
module_files = list(Path(modules_directory).rglob("*.py"))
# Control module
controlspec = spec_from_file_location("control", str([mf for mf in module_files if mf.name == "control.py"][0]))
control = module_from_spec(controlspec)
sys.modules["control"] = control
controlspec.loader.exec_module(control)
# Vision module
visionspec = spec_from_file_location("vision", str([mf for mf in module_files if mf.name == "vision.py"][0]))
vision = module_from_spec(visionspec)
sys.modules["vision"] = vision
visionspec.loader.exec_module(vision)
# Debug module (not to be included in final build)
debugspec = spec_from_file_location("debug", str([mf for mf in module_files if mf.name == "debug.py"][0]))
debug = module_from_spec(debugspec)
sys.modules["debug"] = debug
debugspec.loader.exec_module(debug)

# Functions
def launch_tux_fisher(delay=3):
    # Open Tux Fishing
    os.popen("xdg-open https://pushergames.itch.io/tuxfishing") # opens the web version of Tux Fishing
    time.sleep(delay) # Wait for Tux Fishing to open
    # fullscreen the game by finding the fullscreen button and clicking it
    fullscreen_button = [img for img in image_files if img.name == "fullscreen_icon.png"][0]
    if fullscreen_button:
        try:
            fullscreen_location = pyautogui.locateCenterOnScreen(str(fullscreen_button), confidence=0.8)
            if fullscreen_location:
                pyautogui.click(fullscreen_location)
                time.sleep(2) # Wait for the game to go fullscreen
            else:
                print("Fullscreen button not found on screen.")
        except Exception as e:
            print(f"Error locating fullscreen button: {e}")
    else:
        print("Fullscreen button image not found in images directory.")
def debugORB():
    # Launch Tux Fisher
    launch_tux_fisher()
    # Debug ORB keypoints
    debug.debug_orb_keypoints()

# Main Logic
def main():
    # Launch Tux Fisher
    launch_tux_fisher()
    # run for 30 seconds and locate splashes
    start_time = time.time()
    while time.time() - start_time < 30:
        # Locate splashes using ORB
        points = vision.locate_splashes_orb(roi=(0, 300, 2600, 300))
        if points:
            for point in points:
                print(f"Splash found at {point} using ORB")
        else:
            print("No splashes found using ORB")
        time.sleep(0.1) # Wait before next search



if __name__ == "__main__":
    # main()
    debugORB()