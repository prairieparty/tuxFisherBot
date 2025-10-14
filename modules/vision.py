import cv2 as cv
from pathlib import Path
import numpy as np
import pyautogui
from mss import mss
import math
from time import sleep

# Module to handle vision-related tasks
def screenshot_roi(roi=None, monitor_index=1):
    """
    Capture a screenshot of the whole screen or a specific ROI.

    Args:
        roi (tuple): (x, y, w, h) region of interest in absolute screen coords.
        monitor_index (int): Index of the monitor to capture (1 = primary).

    Returns:
        np.ndarray: BGR image from the captured region.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        mon_left, mon_top = monitor["left"], monitor["top"]
        mon_width, mon_height = monitor["width"], monitor["height"]

        if roi:
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


def screenshot_mask(mode="black", roi=None, monitor_index=1):
    """
    Capture a screenshot and return a binary mask of black or white areas.

    Args:
        mode (str): "black" or "white".
        roi (tuple): (x, y, w, h) region of interest.
        monitor_index (int): Monitor index (1 = primary).

    Returns:
        np.ndarray: Binary mask (255 where matched, 0 elsewhere).
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        mon_left, mon_top = monitor["left"], monitor["top"]
        mon_width, mon_height = monitor["width"], monitor["height"]

        if roi:
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

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        if mode == "black":
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 35])  # tweak brightness cutoff as needed
            mask = cv.inRange(hsv, lower_black, upper_black)
        elif mode == "white":
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 40, 255])
            mask = cv.inRange(hsv, lower_white, upper_white)
        else:
            raise ValueError("Invalid mode. Use 'black' or 'white'.")

        return mask

class RodAngleTracker:
    def __init__(self, alpha=0.2):
        """
        alpha = smoothing factor (0 < alpha ≤ 1)
        smaller alpha = smoother but more lag
        """
        self.alpha = alpha
        self.smoothed_angle = None

    def determine_avatar_angle(self, frame):
        """
        Detect penguin center and fishing rod tip, then calculate smoothed facing angle.
        Returns (smoothed_angle, raw_angle, penguin_center, rod_tip)
        """
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --- Detect penguin (white belly) ---
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 60, 255])
        penguin_mask = cv.inRange(hsv, lower_white, upper_white)
        penguin_mask = cv.morphologyEx(penguin_mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        penguin_mask = cv.morphologyEx(penguin_mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv.findContours(penguin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.smoothed_angle, None, None, None

        penguin_contour = max(contours, key=cv.contourArea)
        M = cv.moments(penguin_contour)
        if M["m00"] == 0:
            return self.smoothed_angle, None, None, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        penguin_center = (cx, cy)

        # --- Detect fishing rod ---
        lower_rod = np.array([10, 60, 40])
        upper_rod = np.array([30, 255, 255])
        rod_mask = cv.inRange(hsv, lower_rod, upper_rod)
        rod_mask = cv.morphologyEx(rod_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        rod_mask = cv.morphologyEx(rod_mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv.findContours(rod_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.smoothed_angle, None, penguin_center, None

        rod_contour = max(contours, key=cv.contourArea)
        rod_tip = tuple(rod_contour[rod_contour[:, :, 1].argmin()][0])

        # --- Compute raw angle ---
        dx = rod_tip[0] - penguin_center[0]
        dy = penguin_center[1] - rod_tip[1]
        raw_angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        # --- Smooth angle ---
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            # handle wrap-around near 0/360 to avoid 359→1 jumps
            delta = ((raw_angle - self.smoothed_angle + 540) % 360) - 180
            self.smoothed_angle = (self.smoothed_angle + self.alpha * delta) % 360

        return self.smoothed_angle, raw_angle, penguin_center, rod_tip
    
def determine_avatar_angle():
    '''Determine the angle of the avatar based on the white areas in the mask, in degrees.
    Returns the degree the avatar is facing.'''
    mask = screenshot_mask(mode="white")
    h, w = mask.shape
    left_half = mask[:, :w//2]
    right_half = mask[:, w//2:]

    left_white = cv.countNonZero(left_half)
    right_white = cv.countNonZero(right_half)
    total_white = left_white + right_white

    if total_white == 0:
        # The penguin is facing completely away (no white detected)
        return 180
    else:
        # Calculate the angle based on the ratio of white areas
        angle = (right_white - left_white) / total_white * 90  # Scale to -90 to +90 degrees
        return angle

def locate_splashes_orb(image_path=None, roi=(400, 300, 600, 200), threshold=10):
    """
    Detect activity (fish/splash) in a region of the screen using ORB keypoints.

    Args:
        image_path (str): Optional template path (kept for compatibility, not used here).
        roi (tuple): (x, y, w, h) bounding box region of interest.
        threshold (int): Number of keypoints inside ROI needed to confirm detection.

    Returns:
        list of (x, y): List with center coordinates of detected activity (empty if none).
    """
    # Take screenshot
    screenshot = pyautogui.screenshot()
    frame = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # ORB detector (high sensitivity)
    orb = cv.ORB_create(
        nfeatures=3000,
        scaleFactor=1.1,
        nlevels=12,
        edgeThreshold=15,
        fastThreshold=10
    )
    kp, des = orb.detectAndCompute(gray, None)

    if not kp:
        return []

    x, y, w, h = roi

    # Count points inside ROI
    points_in_roi = [(int(p.pt[0]), int(p.pt[1])) for p in kp
                     if x <= p.pt[0] <= x+w and y <= p.pt[1] <= y+h]

    if len(points_in_roi) >= threshold:
        # Return center of ROI as detection location
        cx, cy = x + w // 2, y + h // 2
        return [(cx, cy)]
    else:
        return []

def debug_orb_keypoints(image_path):
    """
    Debug function to visualize ORB keypoints on the splash image
    and the current game screenshot.
    """

    # Take screenshot
    screenshot = pyautogui.screenshot()
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    screenshot_gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

    # Load splash image
    splash_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if splash_image is None:
        raise FileNotFoundError(f"Splash image not found at {image_path}")

    # ORB detector
    orb = cv.ORB_create()

    # Detect keypoints
    kp1, des1 = orb.detectAndCompute(splash_image, None)
    kp2, des2 = orb.detectAndCompute(screenshot_gray, None)

    # Draw keypoints
    splash_kp = cv.drawKeypoints(splash_image, kp1, None, color=(0,255,0), flags=0)
    screenshot_kp = cv.drawKeypoints(screenshot_gray, kp2, None, color=(0,255,0), flags=0)

    # Show windows
    cv.imshow("Splash Keypoints", splash_kp)
    cv.imshow("Screenshot Keypoints", screenshot_kp)
    cv.waitKey(0)
    cv.destroyAllWindows()

def live_orb_debug(image_path, delay=0.1):
    """
    Continuously show ORB keypoints over the game screenshot.
    
    Args:
        image_path (str): Path to splash template.
        delay (float): Delay between frames in seconds.
    """

    # Load splash template
    splash_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if splash_image is None:
        raise FileNotFoundError(f"Splash image not found at {image_path}")

    orb = cv.ORB_create()

    # Detect splash keypoints once
    kp1, des1 = orb.detectAndCompute(splash_image, None)
    splash_with_kp = cv.drawKeypoints(
        splash_image, kp1, None, color=(255, 0, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    while True:
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        screenshot_gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

        # Detect screenshot keypoints
        kp2, des2 = orb.detectAndCompute(screenshot_gray, None)

        # Draw keypoints directly on screenshot
        screenshot_with_kp = cv.drawKeypoints(
            screenshot, kp2, None, color=(0, 255, 0),
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Overlay template in top-left for reference
        h, w = splash_with_kp.shape[:2]
        screenshot_with_kp[0:h, 0:w] = cv.cvtColor(splash_with_kp, cv.COLOR_BGR2RGB)

        # Show
        cv.imshow("Live ORB Keypoints", screenshot_with_kp)

        # Exit on "q"
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        sleep(delay)

    cv.destroyAllWindows()

def locate_splashes_cv(image_path, threshold=0.8):
    """
    Locate all instances of the splash image on the screen.
    
    Args:
        image_path (str): Path to the splash image file.
        threshold (float): Matching threshold (default is 0.8).
        
    Returns:
        list of tuples: List of (x, y) coordinates where the splash is found.
    """
    # Take a screenshot
    screenshot = pyautogui.screenshot()
    #save the screenshot for debugging
    screenshot.save("screenshot.png")
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    
    # Load the splash image
    splash_image = cv.imread(image_path)
    if splash_image is None:
        raise FileNotFoundError(f"Splash image not found at {image_path}")
    
    # Perform template matching
    result = cv.matchTemplate(screenshot, splash_image, cv.TM_CCOEFF_NORMED)
    
    # Get locations where the match exceeds the threshold
    locations = np.where(result >= threshold)
    
    # Create a list of (x, y) coordinates
    points = list(zip(locations[1], locations[0]))  # (x, y) format
    
    return points

def locate_splashes_pyautogui(image_path, threshold=0.8):
    """
    Locate all instances of the splash image on the screen using pyautogui.
    
    Args:
        image_path (str): Path to the splash image file.
        threshold (float): Matching threshold (default is 0.8).
        
    Returns:
        list of tuples: List of (x, y) coordinates where the splash is found.
    """
    # Load the splash image
    splash_image = pyautogui.locateAllOnScreen(image_path, confidence=threshold)
    
    if splash_image is None:
        raise FileNotFoundError(f"Splash image not found at {image_path}")
    
    # Create a list of (x, y) coordinates
    points = [(loc.left, loc.top) for loc in splash_image]
    
    return points

def locate_differences(threshold=0.8):
    """
    Locate differences between two screenshots.
    
    Args:
        threshold (float): Matching threshold (default is 0.8).
        
    Returns:
        list of tuples: List of (x, y) coordinates where differences are found.
    """
    # Take two screenshots
    screenshot1 = pyautogui.screenshot()
    sleep(0.5)  # Wait for half a second
    screenshot2 = pyautogui.screenshot()
    
    screenshot1 = cv.cvtColor(np.array(screenshot1), cv.COLOR_RGB2BGR)
    screenshot2 = cv.cvtColor(np.array(screenshot2), cv.COLOR_RGB2BGR)
    
    # Compute the absolute difference between the two screenshots
    diff = cv.absdiff(screenshot1, screenshot2)
    
    # Convert the difference image to grayscale
    gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to get binary image
    _, binary_diff = cv.threshold(gray_diff, int(threshold * 255), 255, cv.THRESH_BINARY)
    
    # Find contours of the differences
    contours, _ = cv.findContours(binary_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    points = []
    for contour in contours:
        if cv.contourArea(contour) > 100:  # Filter out small areas
            x, y, w, h = cv.boundingRect(contour)
            points.append((x + w // 2, y + h // 2))  # Append center of the bounding box
    
    return points

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