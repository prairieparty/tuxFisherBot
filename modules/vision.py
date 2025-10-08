import cv2 as cv
from pathlib import Path
import numpy as np
import pyautogui
from time import sleep

# Module to handle vision-related tasks

def determine_player_angle(avatar_image_front, avatar_image_back, screenshot=None):
    ''' Determine the angle of the player avatar in the screenshot;
    the avatar starts off facing dead on towards the screen (0 degrees).
    Args:
        avatar_image_path (str): Path to the avatar image file.
        screenshot (PIL.Image or None): Optional screenshot to analyze. If None, takes a new screenshot.
    Returns:
        float: Angle in degrees the avatar is facing (0 = straight, positive = right,
                negative = left). Returns None if avatar not found.
    '''
    #Goal is to rotate the penguin 180, so that we can see it from the back.
    #penguin is just a black shape against a blue background, so we can use color thresholding to find it.
    if screenshot is None:
        screenshot = pyautogui.screenshot()
    frame = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Define color range for the penguin (black)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv.inRange(hsv, lower_black, upper_black)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Assume the largest contour is the penguin
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    penguin_roi = frame[y:y+h, x:x+w]
    penguin_gray = cv.cvtColor(penguin_roi, cv.COLOR_BGR2GRAY)
    # Load avatar images
    front_img = cv.imread(avatar_image_front, cv.IMREAD_GRAYSCALE)
    back_img = cv.imread(avatar_image_back, cv.IMREAD_GRAYSCALE)
    if front_img is None or back_img is None:
        raise FileNotFoundError("Avatar image files not found.")
    # Template matching
    res_front = cv.matchTemplate(penguin_gray, front_img, cv.TM_CCOEFF_NORMED)
    res_back = cv.matchTemplate(penguin_gray, back_img, cv.TM_CCOEFF_NORMED)
    _, max_val_front, _, _ = cv.minMaxLoc(res_front)
    _, max_val_back, _, _ = cv.minMaxLoc(res_back)
    # Determine angle based on which template matched better
    if max_val_front > max_val_back and max_val_front > 0.5:
        return 0.0  # Facing front
    elif max_val_back > max_val_front and max_val_back > 0.5:
        return 180.0  # Facing back
    else:
        return None  # Avatar not found or unclear

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