import cv2 as cv
from pathlib import Path
import numpy as np
import pyautogui
from mss import mss
import math
from PyQt5 import QtCore
import time
from sklearn.cluster import DBSCAN
from collections import deque

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
                          self.screen_size[1]//8, 
                          self.screen_size[0], 
                          self.screen_size[1]//4)  # x, y, w, h
        
        self.splashThreshold = 100 # number of keypoints to confirm splash

        self.splashOrb = cv.ORB_create(
            nfeatures=3000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=10
        )

        # splash detector parameters
        self.splashThreshold = 25           # lowered from 60 → more sensitive
        self.splash_alpha = 0.45            # faster angular response
        self.splash_point_beta = 0.55       # faster position tracking
        self.splash_angle_smooth = None
        self.splash_point_smooth = None

        self._kp_hist = deque(maxlen=20)
        self._last_detect_t = 0.0
        self._cooldown_sec = 0.15          # shorter cooldown (was 0.3)
        self._need_confirm = False
        self._pending_pt = None
        self.motion_detection_frames = [] # will hold current frame and one prior for motion detection

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
        self.whitePercentageThreshold = 0.04 # threshold for white belly detection

    def _circular_ema(self, prev_deg, new_deg, alpha):
        if prev_deg is None:
            return new_deg
        diff = ((new_deg - prev_deg + 540) % 360) - 180
        return (prev_deg + alpha * diff) % 360

    def _robust_centroid(self, xs, ys):
        # remove outliers with MAD (median absolute deviation)
        x = np.asarray(xs); y = np.asarray(ys)
        mx, my = np.median(x), np.median(y)
        madx = np.median(np.abs(x - mx)) + 1e-6
        mady = np.median(np.abs(y - my)) + 1e-6
        keep = (np.abs(x - mx) <= 2.5 * madx) & (np.abs(y - my) <= 2.5 * mady)
        xk, yk = x[keep], y[keep]
        if xk.size < 5:   # fall back if we pruned too hard
            xk, yk = x, y
        # median is stabler than mean for splashes
        return int(np.median(xk)), int(np.median(yk))

    def update_splash_detector(self):
        """
        Detect splash using motion (frame differencing), while excluding
        sky and player zones. Returns (screen_point, smoothed_angle_deg) or None.
        """
        x, y, w, h = self.splashROI

        # Capture ROI only
        with mss() as sct:
            roi_img = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h}))
        frame = cv.cvtColor(roi_img, cv.COLOR_BGRA2BGR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Initialize history
        if not hasattr(self, "_prev_splash_gray"):
            self._prev_splash_gray = gray
            return None

        # --- 1. Motion difference ---
        diff = cv.absdiff(gray, self._prev_splash_gray)
        _, motion_mask = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

        # --- 2. Exclusion masks ---

        # 2a. Exclude everything above splash ROI (already handled by ROI crop)
        # 2b. Exclude slightly expanded player zone
        px, py, pw, ph = self.penguinROI
        expand = 40  # expand the player exclusion box
        ex_left   = max(0, px - expand - x)
        ex_top    = max(0, py - expand - y)
        ex_right  = min(w, px + pw + expand - x)
        ex_bottom = min(h, py + ph + expand - y)
        cv.rectangle(motion_mask, (ex_left, ex_top), (ex_right, ex_bottom), 0, -1)

        # --- 3. Morphological cleanup ---
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, kernel)

        # --- 4. Find motion blobs ---
        contours, _ = cv.findContours(motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.motion_contours = contours  # for debugging/visualization
        if not contours:
            self._prev_splash_gray = gray
            self.motion_contours = []
            return None

        # Choose largest motion blob
        best_c = max(contours, key=cv.contourArea)
        area = cv.contourArea(best_c)
        if area < 80 or area > 20000:
            self._prev_splash_gray = gray
            return None

        # --- 5. Compute centroid ---
        M = cv.moments(best_c)
        if M["m00"] == 0:
            self._prev_splash_gray = gray
            return None
        cx_local = int(M["m10"] / M["m00"])
        cy_local = int(M["m01"] / M["m00"])

        # --- 6. Convert to screen coords ---
        sx, sy = x + cx_local, y + cy_local

        # --- 7. Smooth splash point ---
        if self.splash_point_smooth is None:
            self.splash_point_smooth = (float(sx), float(sy))
        else:
            px_s, py_s = self.splash_point_smooth
            px_s = (1 - self.splash_point_beta) * px_s + self.splash_point_beta * sx
            py_s = (1 - self.splash_point_beta) * py_s + self.splash_point_beta * sy
            self.splash_point_smooth = (px_s, py_s)
        sx_s, sy_s = self.splash_point_smooth

        # --- 8. Compute world angle relative to player ---
        if hasattr(self, "penguin_center_screen") and self.penguin_center_screen is not None:
            ox, oy = self.penguin_center_screen
        else:
            ox, oy = self.screen_center

        dx = sx_s - ox
        dy = sy_s - oy
        if dx * dx + dy * dy < 25 * 25:
            self._prev_splash_gray = gray
            return None

        raw_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
        ang = self._circular_ema(self.splash_angle_smooth, raw_angle, self.splash_alpha)
        self.splash_angle_smooth = ang
        self.last_splash_point = (int(round(sx_s)), int(round(sy_s)))

        # Update previous frame
        self._prev_splash_gray = gray

        if self.debug:
            print(f"[MotionSplash] at ({sx_s:.1f},{sy_s:.1f}) → {self.splash_angle_smooth:.1f}° area={area:.0f}")

        return (self.last_splash_point, self.splash_angle_smooth)

    def motion_detection(self, bbox_thresh=400, nms_thresh=1e-3, frames=None):
        '''An alternate method for detecting splashes using motion detection.'''
        ''' if frames is provided, use those frames instead of taking new screenshots '''
        
        # helper functions
        def get_mask(frame1, frame2, kernel=np.array((9,9), dtype=np.uint8)):
            """ Obtains image mask
                Inputs: 
                    frame1 - Grayscale frame at time t
                    frame2 - Grayscale frame at time t + 1
                    kernel - (NxN) array for Morphological Operations
                Outputs: 
                    mask - Thresholded mask for moving pixels
                """

            frame_diff = cv.subtract(frame2, frame1)

            # blur the frame difference
            frame_diff = cv.medianBlur(frame_diff, 3)
            mask = cv.adaptiveThreshold(frame_diff, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY_INV, 11, 3)

            mask = cv.medianBlur(mask, 3)

            # morphological operations
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

            return mask

        def get_contour_detections(mask, thresh=400):
            """ Obtains initial proposed detections from contours discoverd on the mask. 
                Scores are taken as the bbox area, larger is higher.
                Inputs:
                    mask - thresholded image mask
                    thresh - threshold for contour size
                Outputs:
                    detectons - array of proposed detection bounding boxes and scores [[x1,y1,x2,y2,s]]
                """
            # get mask contours
            contours, _ = cv.findContours(mask, 
                                        cv.RETR_EXTERNAL, # cv.RETR_TREE, 
                                        cv.CHAIN_APPROX_TC89_L1)
            detections = []
            for cnt in contours:
                x,y,w,h = cv.boundingRect(cnt)
                area = w*h
                if area > thresh: 
                    detections.append([x,y,x+w,y+h, area])

            return np.array(detections)
        
        def remove_contained_bboxes(boxes):
            """ Removes all smaller boxes that are contained within larger boxes.
                Requires bboxes to be soirted by area (score)
                Inputs:
                    boxes - array bounding boxes sorted (descending) by area 
                            [[x1,y1,x2,y2]]
                Outputs:
                    keep - indexes of bounding boxes that are not entirely contained 
                        in another box
                """
            check_array = np.array([True, True, False, False])
            keep = list(range(0, len(boxes)))
            for i in keep: # range(0, len(bboxes)):
                for j in range(0, len(boxes)):
                    # check if box j is completely contained in box i
                    if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                        try:
                            keep.remove(j)
                        except ValueError:
                            continue
            return keep

        def non_max_suppression(boxes, scores, threshold=1e-1):
            """
            Perform non-max suppression on a set of bounding boxes and corresponding scores.
            Inputs:
                boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
                scores: a list of corresponding scores 
                threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
            Outputs:
                boxes - non-max suppressed boxes
            """
            # Sort the boxes by score in descending order
            boxes = boxes[np.argsort(scores)[::-1]]

            # remove all contained bounding boxes and get ordered index
            order = remove_contained_bboxes(boxes)

            keep = []
            while order:
                i = order.pop(0)
                keep.append(i)
                for j in order:
                    # Calculate the IoU between the two boxes
                    intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                                max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                    union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                            (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
                    iou = intersection / union

                    # Remove boxes with IoU greater than the threshold
                    if iou > threshold:
                        order.remove(j)
                        
            return boxes[keep]

        # if frames is a list with two frames, use those
        if frames is not None and len(frames) == 2:
            img1, img2 = frames
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        else:
            frame = screenshot_roi(roi=self.splashROI)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.motion_detection_frames.append(gray)
            if len(self.motion_detection_frames) < 2: return None
            if len(self.motion_detection_frames) > 2:
                self.motion_detection_frames.pop(0)

            # convert to grayscale
            img1 = self.motion_detection_frames[0]
            img2 = self.motion_detection_frames[1]

        # compute motion mask
        kernel = np.array((9,9), dtype=np.uint8)
        mask = get_mask(img1, img2, kernel)

        # get initially proposed detections from contours
        detections = get_contour_detections(mask, bbox_thresh)

        # separate bboxes and scores
        bboxes = detections[:, :4]
        scores = detections[:, -1]

        # perform Non-Maximal Supression on initial detections
        nmax = non_max_suppression(bboxes, scores, nms_thresh)

        return nmax

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
        dx = self.rod_tip[0] - self.penguin_center[0]
        dy = self.rod_tip[1] - self.penguin_center[1]

        # raw geometric angle: +x = right (0°), +y = down (so negate dy)
        raw_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

        # --- 5. Mirror if penguin is facing backward (to unify world orientation) ---
        # When facing backward, flip 180° to represent rod direction in world space
        if not self.facing_forward:
            raw_angle = (raw_angle + 180) % 360

        # --- 6. Apply exponential smoothing (to prevent jitter) ---
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            diff = ((raw_angle - self.smoothed_angle + 540) % 360) - 180
            self.smoothed_angle = (self.smoothed_angle + self.penguin_alpha * diff) % 360

        # --- 7. Store the penguin center in screen coordinates for global use ---
        self.penguin_center_screen = (
            self.penguinROI[0] + self.penguin_center[0],
            self.penguinROI[1] + self.penguin_center[1],
        )

        if self.debug:
            facing = 'forward' if self.facing_forward else 'backward'
            print(f"[Penguin] {self.smoothed_angle:.1f}° ({facing})")

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