import cv2 as cv
from pathlib import Path
import numpy as np
import pyautogui
from mss import mss
import math
from PyQt5 import QtCore
from time import sleep

# Module to handle vision-related tasks

def get_center_of_screen(monitor_index=1):
    """
    Get the center coordinates of the specified monitor.

    Args:
        monitor_index (int): Index of the monitor (1 = primary).

    Returns:
        tuple: (center_x, center_y) coordinates of the screen center.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        mon_left, mon_top = monitor["left"], monitor["top"]
        mon_width, mon_height = monitor["width"], monitor["height"]
        center_x = mon_left + mon_width // 2
        center_y = mon_top + mon_height // 2
        return (center_x, center_y)
    
def get_screen_size(monitor_index=1):
    """
    Get the width and height of the specified monitor.

    Args:
        monitor_index (int): Index of the monitor (1 = primary).
    Returns:
        tuple: (width, height) of the screen.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        mon_width, mon_height = monitor["width"], monitor["height"]
        return (mon_width, mon_height)
    
def screenshot_roi(roi=None, monitor_index=1, centered=False):
    """
    Capture a screenshot of the whole screen or a specific ROI.

    Args:
        roi (tuple): (x, y, w, h) region of interest in absolute screen coords.
        monitor_index (int): Index of the monitor to capture (1 = primary).
        centered (bool): Whether to center the ROI around the screen center.

    Returns:
        np.ndarray: BGR image from the captured region.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        mon_left, mon_top = monitor["left"], monitor["top"]
        mon_width, mon_height = monitor["width"], monitor["height"]
        center_x, center_y = get_center_of_screen(monitor_index)

        if roi:
            if centered:
                w, h = roi[0], roi[1]
                x = center_x - w // 2
                y = center_y - h // 2
                roi = (x, y, w, h)
            else:
                x, y, w, h = roi

            # clip ROI to screen boundaries
            left = max(mon_left, x)
            top = max(mon_top, y)
            right = min(mon_left + mon_width, x + w)
            bottom = min(mon_top + mon_height, y + h)
            width = max(0, right - left)
            height = max(0, bottom - top)
            capture_region = {"top": top, "left": left, "width": width, "height": height}
        else:
            capture_region = monitor

        screenshot = sct.grab(capture_region)
        frame = np.array(screenshot)
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        return frame

class VisionCortex():
    """Class to handle vision processing tasks."""
    def __init__(self, debug=False):

        self.screen_center = get_center_of_screen()
        self.screen_size = get_screen_size()

        self.debug = debug

        # initialize splash detector

        self.splashROI = (0, 
                          self.screen_size[1]//4, 
                          self.screen_size[0], 
                          self.screen_size[1]//8)  # x, y, w, h
        
        self.splashThreshold = 15 # number of keypoints to confirm splash

        self.splashOrb = cv.ORB_create(
            nfeatures=3000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=10
        )

        # initialize player angle detector
        pengX = int(self.screen_center[0] - (self.screen_size[0]//6.4) // 2)
        pengY = int(self.screen_center[1] - (self.screen_size[1]//4) // 2)

        self.penguinROI = (pengX,
                           pengY,
                           int(self.screen_size[0]//6.4),
                           int(self.screen_size[1]//4))  # x, y, w, h

        self.penguin_alpha = 0.25 # smoothing factor for angle updates
        self.penguin_center = None # penguin center point, dynamically updated
        self.rod_tip = None # fishing rod tip point, dynamically updated
        self.smoothed_angle = None # smoothed angle value
        self.facing_forward = None # facing direction (True/False)

        # colors for the penguin and rod detection
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 50])
        self.lower_rod = np.array([10, 80, 80])
        self.upper_rod = np.array([35, 255, 255])
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 50, 255])
        self.lower_orange = np.array([5, 150, 150])
        self.upper_orange = np.array([15, 255, 255])
        self.whitePercentageThreshold = 0.03 # threshold for white belly detection

    def update_splash_detector(self):
        """Run splash detection on the current screen and return splash coordinates if detected."""
        # Capture the region of interest (waterline area)
        frame = screenshot_roi(roi=self.splashROI)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect ORB keypoints in the ROI
        kp, des = self.splashOrb.detectAndCompute(gray, None)

        detected = False
        detection_point = None

        if kp:
            x, y, w, h = self.splashROI
            inside = [(p.pt[0], p.pt[1]) for p in kp if x <= p.pt[0] <= x + w and y <= p.pt[1] <= y + h]

            # If we have enough keypoints, mark detection
            if len(inside) >= self.splashThreshold:
                detected = True
                avg_x = int(np.mean([p[0] for p in inside]))
                avg_y = int(np.mean([p[1] for p in inside]))
                detection_point = (avg_x, avg_y + 30)  # small offset below ROI for realism

                # Update persistent state
                self.last_splash_point = detection_point

                if self.debug:
                    print(f"[Splash Detected] {len(inside)} keypoints → ({avg_x}, {avg_y})")

            elif self.debug:
                print(f"[Splash Scan] {len(inside)} keypoints in ROI (threshold: {self.splashThreshold})")

        else:
            if self.debug:
                print("[Splash Scan] No keypoints found in frame.")

        # Return the detection point or None
        return detection_point if detected else None

        
    def update_player_detector(self):
        """Run player angle detection on the current screen."""
        frame = screenshot_roi(roi=self.penguinROI)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --- 1. Detect penguin (largest dark blob near center) ---
        mask_black = cv.inRange(hsv, self.lower_black, self.upper_black)
        mask_black = cv.morphologyEx(mask_black, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_black = cv.morphologyEx(mask_black, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv.findContours(mask_black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            if self.debug: print("No penguin detected.")
            return None

        penguin_contour = max(contours, key=cv.contourArea)
        M = cv.moments(penguin_contour)
        if M["m00"] == 0:
            if self.debug: print("Penguin contour has zero area.")
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        self.penguin_center = (cx, cy)

        # --- 2. Detect fishing rod (light brown/yellow) ---
        mask_rod = cv.inRange(hsv, self.lower_rod, self.upper_rod)
        mask_rod = cv.morphologyEx(mask_rod, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_rod = cv.morphologyEx(mask_rod, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Keep only rod pixels above penguin center
        mask_height, mask_width = mask_rod.shape
        penguin_y = self.penguin_center[1]
        mask_rod[penguin_y:mask_height, :] = 0

        contours, _ = cv.findContours(mask_rod, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            if self.debug: print("No fishing rod detected.")
            return None

        # Find the topmost rod point
        min_y = mask_height
        rod_tip = None
        for c in contours:
            y_min = c[:, :, 1].min()
            if y_min < min_y:
                min_y = y_min
                rod_tip = tuple(c[c[:, :, 1].argmin()][0])
        if rod_tip is None:
            if self.debug: print("No fishing rod tip found.")
            return None
        self.rod_tip = rod_tip

        # --- 3. Determine facing direction using white belly visibility ---
        mask_white = cv.inRange(hsv, self.lower_white, self.upper_white)
        # Focus on lower central third of the ROI
        h, w = mask_white.shape
        roi_center = mask_white[h // 3 : h, w // 3 : 2 * w // 3]
        white_ratio = cv.countNonZero(roi_center) / roi_center.size

        self.facing_forward = white_ratio > self.whitePercentageThreshold

        # --- 4. Compute rod angle relative to penguin center ---
        dx = rod_tip[0] - self.penguin_center[0]
        dy = rod_tip[1] - self.penguin_center[1]
        raw_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

        # Mirror horizontally if penguin is facing away
        if not self.facing_forward:
            raw_angle = (540 - raw_angle) % 360  # mirror across vertical axis

        # --- 5. Apply smoothing to avoid jitter ---
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            delta = ((raw_angle - self.smoothed_angle + 540) % 360) - 180
            self.smoothed_angle = (self.smoothed_angle + self.penguin_alpha * delta) % 360

        # --- 6. Set the angle to a flat 90 if there are even amounts of orange pixels on left and right sides of ROI ---
        mask_orange = cv.inRange(hsv, self.lower_orange, self.upper_orange)
        orange_count = cv.countNonZero(mask_orange)
        if orange_count > 0:
            h, w = mask_orange.shape
            left_half = mask_orange[:, : w // 2]
            right_half = mask_orange[:, w // 2 :]
            left_count = cv.countNonZero(left_half)
            right_count = cv.countNonZero(right_half)
            if abs(left_count - right_count) / orange_count < 0.1:  # within 10%
                self.smoothed_angle = 90.0

        if self.debug: print(f"Angle: {self.smoothed_angle:.1f}°, facing {'forward' if self.facing_forward else 'backward'}.")
        return self.smoothed_angle, self.facing_forward

def locate_fullscreen(image_path, threshold=0.8):
    """
    Locate the full screen image on the screen.
    
    Args:
        image_path (str): Path to the full screen image file.
        threshold (float): Matching threshold (default is 0.8).
        
    Returns:
        tuple: (x, y) coordinates where the full screen image is found, or None if not found.
    """
    # Take a screenshot
    screenshot = pyautogui.screenshot()
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    
    # Load the full screen image
    full_image = cv.imread(image_path)
    if full_image is None:
        raise FileNotFoundError(f"Full screen image not found at {image_path}")
    
    # Perform template matching
    result = cv.matchTemplate(screenshot, full_image, cv.TM_CCOEFF_NORMED)
    
    # Get the best match location
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    
    if max_val >= threshold:
        return max_loc  # (x, y) format
    else:
        return None