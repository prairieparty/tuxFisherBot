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

        self.splashThreshold = 60          # base minimum count (you can tune)
        self.splash_alpha = 0.35           # angle EMA
        self.splash_point_beta = 0.4       # point EMA
        self.splash_angle_smooth = None
        self.splash_point_smooth = None

        # Anti-FP state
        self._kp_hist = deque(maxlen=30)   # rolling history of kp counts
        self._last_detect_t = 0.0
        self._cooldown_sec = 0.35          # refractory period after a detection
        self._need_confirm = False         # 2-frame confirmation
        self._pending_pt = None

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
        Robust splash detection.
        Returns (screen_point, smoothed_angle_deg) or None.
        """
        x, y, w, h = self.splashROI

        # 1) Capture ROI ONLY (prevents overlay/sky noise)
        with mss() as sct:
            roi_img = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h}))
        frame = cv.cvtColor(roi_img, cv.COLOR_BGRA2BGR)
        gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 2) ORB keypoints inside ROI
        kps, des = self.splashOrb.detectAndCompute(gray, None)
        kp_count = 0 if not kps else len(kps)
        self._kp_hist.append(kp_count)

        # 3) Rolling baseline spike test (mean + 3σ) to reject “always busy” water
        if len(self._kp_hist) >= 10:
            mu = np.mean(self._kp_hist)
            sd = np.std(self._kp_hist) + 1e-6
            spike_ok = kp_count >= max(self.splashThreshold, mu + 3.0 * sd)
        else:
            spike_ok = kp_count >= self.splashThreshold

        if not kps or not spike_ok:
            self._need_confirm = False
            return None

        # 4) Cluster keypoints — we want one dense splash, not scattered texture
        pts = np.float32([kp.pt for kp in kps])  # (x,y) in ROI coords
        if len(pts) < 8:
            self._need_confirm = False
            return None

        # eps ~20px works well @1080p; tune if needed
        db = DBSCAN(eps=20, min_samples=8).fit(pts)
        labels = db.labels_
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if unique.size == 0:
            self._need_confirm = False
            return None

        # choose largest cluster
        best_label = unique[np.argmax(counts)]
        cluster = pts[labels == best_label]
        cluster_ratio = len(cluster) / max(len(pts), 1)

        # Tightness gate: cluster must be compact
        cx_local, cy_local = np.median(cluster[:, 0]), np.median(cluster[:, 1])
        rad = np.median(np.linalg.norm(cluster - np.array([cx_local, cy_local]), axis=1))
        if cluster_ratio < 0.55 or rad > 45:  # require dense & not too spread out
            self._need_confirm = False
            return None

        # 5) Foam/brightness cue near centroid (splash is bright & low saturation)
        cx_i, cy_i = int(round(cx_local)), int(round(cy_local))
        r = 16
        x1 = max(0, cx_i - r); x2 = min(w, cx_i + r)
        y1 = max(0, cy_i - r); y2 = min(h, cy_i + r)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            self._need_confirm = False
            return None
        hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
        # "white-ish" foam: high V, low-to-mid S
        v_mean = float(hsv[:, :, 2].mean())
        s_mean = float(hsv[:, :, 1].mean())
        foam_ok = (v_mean >= 180) and (s_mean <= 100)

        # Gradient cue (optional): splashes have edges
        sob = cv.Sobel(cv.cvtColor(patch, cv.COLOR_BGR2GRAY), cv.CV_32F, 1, 1, ksize=3)
        grad_ok = float(np.mean(np.abs(sob))) >= 25

        if not (foam_ok or grad_ok):
            self._need_confirm = False
            return None

        # 6) Two-frame confirmation + cooldown (reduces one-off sparkles)
        now = time.monotonic()
        if now - self._last_detect_t < self._cooldown_sec:
            return None

        if not self._need_confirm:
            self._need_confirm = True
            self._pending_pt = (cx_local, cy_local)
            return None

        self._need_confirm = False
        self._last_detect_t = now

        # 7) Convert to screen coords
        sx, sy = x + cx_local, y + cy_local

        # 8) Smooth the point (helps angle stability)
        if self.splash_point_smooth is None:
            self.splash_point_smooth = (float(sx), float(sy))
        else:
            px, py = self.splash_point_smooth
            px = (1 - self.splash_point_beta) * px + self.splash_point_beta * sx
            py = (1 - self.splash_point_beta) * py + self.splash_point_beta * sy
            self.splash_point_smooth = (px, py)
        sx_s, sy_s = self.splash_point_smooth

        # 9) Use penguin center as origin (global coords)
        if hasattr(self, "penguin_center_screen") and self.penguin_center_screen is not None:
            ox, oy = self.penguin_center_screen
        else:
            ox, oy = self.screen_center

        dx = sx_s - ox
        dy = sy_s - oy

        if dx * dx + dy * dy < 40 * 40:
            return None

        # Compute raw world angle (same convention as penguin)
        raw_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

        # Smooth it
        ang = self._circular_ema(self.splash_angle_smooth, raw_angle, self.splash_alpha)
        self.splash_angle_smooth = ang
        self.last_splash_point = (int(round(sx_s)), int(round(sy_s)))

        if self.debug:
            print(f'Final angle: {self.splash_angle_smooth:.1f}° at point ({sx_s:.1f}, {sy_s:.1f})')
            
        return (self.last_splash_point, self.splash_angle_smooth)

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