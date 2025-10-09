import cv2 as cv
from pathlib import Path
import numpy as np
import pyautogui
from mss import mss
from time import sleep

# Module to handle vision-related tasks
def screenshot_roi(roi=None):
    '''Capture a screenshot of the whole screen by default, or a specific ROI if provided.'''
    with mss() as sct:
        # Use the first monitor (index 1); adjust if needed
        monitor = sct.monitors[1]
        if roi:
            x, y, w, h = roi
            monitor = {"top": y, "left": x, "width": w, "height": h}
        screenshot = sct.grab(monitor)
        # Convert to numpy array and BGR for OpenCV
        screen_frame = np.array(screenshot)
        screen_frame = cv.cvtColor(screen_frame, cv.COLOR_BGRA2BGR)
    return screen_frame

def determine_player_angle(avatar_image_front, avatar_image_back, avatar_image_left, avatar_image_right, screenshot=None):
    """
    Determine the angle of the player avatar using ORB feature matching.

    Args:
        avatar_image_front (str): Path to the front view image of the avatar.
        avatar_image_back (str): Path to the back view image of the avatar.
        avatar_image_left (str): Path to the left view image of the avatar.
        avatar_image_right (str): Path to the right view image of the avatar.
        screenshot (numpy array): Optional screenshot image. If None, a new screenshot will be taken.

    Returns:
        float: Angle in degrees (0 = front, 90 = right, 180 = back, 270 = left), or None if not found.
    """
    # Load avatar images
    img_front = cv.imread(avatar_image_front, cv.IMREAD_GRAYSCALE)
    img_back = cv.imread(avatar_image_back, cv.IMREAD_GRAYSCALE)
    img_left = cv.imread(avatar_image_left, cv.IMREAD_GRAYSCALE)
    img_right = cv.imread(avatar_image_right, cv.IMREAD_GRAYSCALE)

    if img_front is None or img_back is None or img_left is None or img_right is None:
        raise ValueError("One or more avatar images not found or could not be loaded.")

    # Take screenshot if not provided
    if screenshot is None:
        screenshot = pyautogui.screenshot()
        screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    gray_screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        fastThreshold=20
    )

    # Detect keypoints and descriptors
    kp_front, des_front = orb.detectAndCompute(img_front, None)
    kp_back, des_back = orb.detectAndCompute(img_back, None)
    kp_left, des_left = orb.detectAndCompute(img_left, None)
    kp_right, des_right = orb.detectAndCompute(img_right, None)
    kp_screenshot, des_screenshot = orb.detectAndCompute(gray_screenshot, None)

    if des_screenshot is None:
        return None

    # BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches_front = bf.match(des_front, des_screenshot)
    matches_back = bf.match(des_back, des_screenshot)
    matches_left = bf.match(des_left, des_screenshot)
    matches_right = bf.match(des_right, des_screenshot)

    # Determine which has more good matches
    num_matches = {
        "front": len(matches_front),
        "back": len(matches_back),
        "left": len(matches_left),
        "right": len(matches_right)
    }
    max_side = max(num_matches, key=num_matches.get)

    if num_matches[max_side] == 0:
        return None

    # Map sides to angles
    angle_map = {
        "front": 0,
        "right": 90,
        "back": 180,
        "left": 270
    }

    return angle_map[max_side]

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