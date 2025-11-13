import cv2 as cv
from pathlib import Path
import numpy as np
import pyautogui
from mss import mss
import math
from PyQt5 import QtCore
import time
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
                          self.screen_size[1]//6, 
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
        self.splashThreshold = 400            # bounding box area threshold
        self.splash_alpha = 0.45              # faster angular response
        self.splash_point_beta = 0.55         # faster position tracking
        self.splash_angle_smooth = None
        self.splash_point_smooth = None

        self._kp_hist = deque(maxlen=20)
        self._last_detect_t = 0.0
        self._cooldown_sec = 0.15          # shorter cooldown (was 0.3)
        self._need_confirm = False
        self._pending_pt = None
        self.motion_detection_frames = [] # will hold current frame and one prior for motion detection

        # --- Motion detection tuning (to suppress horizon waves) ---
        self.motion_diff_thresh = 25          # binary threshold on frame diff (increase to ignore low motion)
        self.horizon_mask_frac = 0.33         # top fraction of splash ROI to ignore (waves/horizon)
        self.min_bbox_h = 16                  # reject very flat bands (min bbox height in px)
        self.max_wave_aspect = 4.0            # reject very wide bands (w/h larger than this)

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
        self.whitePercentageThreshold = 0.035  # slightly lower
        self.rod_band_below = 40              # allow more rod pixels below center

        # widen ROI and relax rod color so it’s visible across headings
        self.penguinROI = (
            int(self.screen_center[0] - (self.screen_size[0]//5.2) // 2),
            int(self.screen_center[1] - (self.screen_size[1]//3.2) // 2),
            int(self.screen_size[0]//5.2),
            int(self.screen_size[1]//3.2)
        )
        # rod color: lower saturation/value to catch dim/edge-on views
        self.lower_rod = np.array([8, 30, 30])
        self.upper_rod = np.array([35, 255, 255])
        self.rod_band_below = 70  # keep more pixels below center to avoid cropping

        # widen ROI a bit to avoid cropping beak/rod at extreme headings
        self.penguinROI = (
            int(self.screen_center[0] - (self.screen_size[0]//4.6) // 2),
            int(self.screen_center[1] - (self.screen_size[1]//2.8) // 2),
            int(self.screen_size[0]//4.6),
            int(self.screen_size[1]//2.8)
        )
        # relax orange (beak) thresholds to survive dim/edge-on views
        self.lower_orange = np.array([5, 90, 90])
        self.upper_orange = np.array([22, 255, 255])
        # keep more pixels below center for rod to avoid blind band
        self.rod_band_below = 70

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

    def motion_detection(self, bbox_thresh=None, nms_thresh=1e-3, frames=None, db=False):
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

            # frame difference
            frame_diff = cv.absdiff(frame2, frame1)

            # slight denoise
            frame_diff = cv.medianBlur(frame_diff, 3)

            # binary threshold (non-inverted) to suppress low-amplitude waves
            _, mask = cv.threshold(frame_diff, self.motion_diff_thresh, 255, cv.THRESH_BINARY)

            # morphology: prefer vertical features, break long horizontal bands
            open_k = cv.getStructuringElement(cv.MORPH_RECT, (3, 7))   # taller kernel
            close_k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_k)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_k)

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
                # reject very flat/wide bands near horizon
                if area <= thresh: 
                    continue
                if h < self.min_bbox_h:
                    continue
                if w / max(h, 1) > self.max_wave_aspect:
                    continue
                detections.append([x,y,x+w,y+h, area])

            # Ensure a 2D array even when empty
            if len(detections) == 0:
                return np.empty((0, 5), dtype=np.int32)
            return np.array(detections, dtype=np.int32)
        
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
        
        def boxes_to_points(boxes):
            """ Converts bounding boxes to center points.
                Inputs:
                    boxes - array of bounding boxes [[x1,y1,x2,y2]]
                Outputs:
                    points - array of center points [[x,y]]
                """
            points = []
            for box in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                points.append([cx, cy])
            return points
        
        if bbox_thresh is None:
            bbox_thresh = self.splashThreshold

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

        # suppress top band within splash ROI (horizon region)
        if mask is not None and mask.size > 0:
            top_cut = int(mask.shape[0] * self.horizon_mask_frac)
            if top_cut > 0:
                mask[:top_cut, :] = 0

        # get initially proposed detections from contours
        detections = get_contour_detections(mask, bbox_thresh)

        # Early out if no detections
        if detections is None or detections.shape[0] == 0:
            return np.empty((0, 4), dtype=np.int32)

        # separate bboxes and scores
        bboxes = detections[:, :4]
        scores = detections[:, -1]

        # perform Non-Maximal Supression on initial detections
        nmax = non_max_suppression(bboxes, scores, nms_thresh)

        if db: return nmax
        # convert final bounding boxes to center points
        points = boxes_to_points(nmax)
        return points
    
    def update_player_detector(self, frame=None, smooth_window=5, return_forward=False):
        """
        Detect penguin + rod tip; return smoothed world angle (and facing flag if requested).
        """
        # acquire frame
        if frame is None:
            ss = np.array(pyautogui.screenshot())
            frame = cv.cvtColor(ss, cv.COLOR_RGB2BGR)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --- Penguin (largest dark blob in penguinROI) ---
        x,y,w,h = self.penguinROI
        sub = frame[y:y+h, x:x+w]
        hsv_sub = cv.cvtColor(sub, cv.COLOR_BGR2HSV)
        mask_black = cv.inRange(hsv_sub, self.lower_black, self.upper_black)
        mask_black = cv.morphologyEx(mask_black, cv.MORPH_CLOSE, np.ones((7,7), np.uint8))
        mask_black = cv.morphologyEx(mask_black, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv.findContours(mask_black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None if not return_forward else (None, None)
        penguin_contour = max(contours, key=cv.contourArea)
        M = cv.moments(penguin_contour)
        if M["m00"] == 0:
            return None if not return_forward else (None, None)
        cx_local = int(M["m10"] / M["m00"])
        cy_local = int(M["m01"] / M["m00"])
        self.penguin_center = (cx_local, cy_local)
        self.penguin_center_screen = (x + cx_local, y + cy_local)

        # --- Rod detection (light brown/yellow) ---
        mask_rod = cv.inRange(hsv_sub, self.lower_rod, self.upper_rod)
        mask_rod = cv.morphologyEx(mask_rod, cv.MORPH_OPEN, np.ones((2,2), np.uint8))
        mask_rod = cv.morphologyEx(mask_rod, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))
        band = getattr(self, "rod_band_below", 70)
        cutoff = min(cy_local + int(band), mask_rod.shape[0] - 1)
        mask_rod[cutoff:, :] = 0
        rod_contours, _ = cv.findContours(mask_rod, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rod_tip = None
        if rod_contours:
            # farthest point from penguin center is robust across headings
            best_pt, best_d2 = None, -1.0
            for c in rod_contours:
                pts = c.reshape(-1, 2)
                dxs = pts[:, 0] - cx_local
                dys = pts[:, 1] - cy_local
                d2 = dxs*dxs + dys*dys
                j = int(np.argmax(d2))
                if float(d2[j]) > best_d2:
                    best_d2 = float(d2[j]); best_pt = (int(pts[j,0]), int(pts[j,1]))
            rod_tip = best_pt

            # --- NEW: robust rod orientation via line fit on largest rod contour ---
            try:
                rc_max = max(rod_contours, key=cv.contourArea)
                pts = rc_max.reshape(-1, 2).astype(np.float32)
                rod_line_angle = None
                rod_line_conf = 0.0
                if pts.shape[0] >= 5:
                    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01)
                    vx, vy = float(vx), float(vy)
                    nrm = math.hypot(vx, vy) + 1e-6
                    vx, vy = vx/nrm, vy/nrm

                    # perpendicular dispersion as quality metric
                    x0, y0 = float(x0), float(y0)
                    d_perp = np.abs((pts[:,0] - x0) * vy - (pts[:,1] - y0) * vx)
                    std_perp = float(np.std(d_perp))
                    # map dispersion to [0..1] (lower dispersion → higher confidence)
                    rod_line_conf = max(0.0, min(1.0, 1.0 / (1.0 + std_perp / 3.0)))

                    # choose sign so it points away from penguin center
                    # pick farthest point along the line direction relative to penguin centroid
                    t_vals = (pts[:,0] - cx_local) * vx + (pts[:,1] - cy_local) * vy
                    if np.mean(t_vals) < 0:
                        vx, vy = -vx, -vy

                    rod_line_angle = (math.degrees(math.atan2(-vy, vx)) + 360.0) % 360.0

                    # optionally update rod_tip to the extreme along the fitted line
                    j = int(np.argmax(np.abs(t_vals)))
                    rod_tip = (int(pts[j,0]), int(pts[j,1]))
            except Exception:
                rod_line_angle = None
                rod_line_conf = 0.0

        self.rod_tip = rod_tip

        # --- Beak (orange) detection ---
        beak_center = None
        mask_orange = cv.inRange(hsv_sub, self.lower_orange, self.upper_orange)
        mask_orange = cv.morphologyEx(mask_orange, cv.MORPH_OPEN, np.ones((2,2), np.uint8))
        orange_cnts, _ = cv.findContours(mask_orange, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if orange_cnts:
            c = max(orange_cnts, key=cv.contourArea)
            if cv.contourArea(c) >= 5:
                m2 = cv.moments(c)
                if m2["m00"] != 0:
                    bx = int(m2["m10"]/m2["m00"]); by = int(m2["m01"]/m2["m00"])
                    beak_center = (bx, by)

        # --- Belly (hysteresis) (unchanged) ---
        mask_white = cv.inRange(hsv_sub, self.lower_white, self.upper_white)
        h_sub, w_sub = mask_white.shape
        belly_roi = mask_white[h_sub//3:h_sub, w_sub//3:2*w_sub//3]
        white_ratio = cv.countNonZero(belly_roi) / max(belly_roi.size, 1)
        if not hasattr(self, "white_hi"):
            base = getattr(self, "whitePercentageThreshold", 0.035)
            self.white_hi = base * 1.00
            self.white_lo = base * 0.60
        if not hasattr(self, "_facing_forward"): self._facing_forward = True
        if white_ratio >= self.white_hi: self._facing_forward = True
        elif white_ratio <= self.white_lo: self._facing_forward = False
        self.facing_forward = self._facing_forward

        # --- Beak (orange) detection to orient axis ---
        beak_center = None
        mask_orange = cv.inRange(hsv_sub, self.lower_orange, self.upper_orange)
        mask_orange = cv.morphologyEx(mask_orange, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
        orange_cnts, _ = cv.findContours(mask_orange, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if orange_cnts:
            c = max(orange_cnts, key=cv.contourArea)
            if cv.contourArea(c) >= 6:
                m2 = cv.moments(c)
                if m2["m00"] != 0:
                    bx = int(m2["m10"]/m2["m00"]); by = int(m2["m01"]/m2["m00"])
                    beak_center = (bx, by)

        # --- Body axis via PCA (undirected) ---
        pts = penguin_contour.reshape(-1, 2).astype(np.float32)
        _, evecs, evals = cv.PCACompute2(pts, mean=None)
        axis = evecs[0]
        # body eccentricity (ratio major/minor) to judge how directional the silhouette is
        ecc = float(evals[0] / (evals[1] + 1e-6))

        # measure rod vector length (0 if missing)
        rod_len = 0.0
        if rod_tip is not None:
            dx_r = rod_tip[0] - cx_local
            dy_r = rod_tip[1] - cy_local
            rod_len = math.hypot(dx_r, dy_r)

        low_info = (rod_len < 12) and (beak_center is None) and (ecc < 2.2)

        # continuity helpers
        if not hasattr(self, "_last_time"):
            self._last_time = time.time()
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        if not hasattr(self, "_last_raw_angle"):
            self._last_raw_angle = None
        if not hasattr(self, "_spin_rate_deg_s"):
            self._spin_rate_deg_s = 0.0

        # --- NEW: confidence-weighted angle fusion (rod tip, rod line, beak, PCA axis) ---
        cand_angles = []
        cand_weights = []

        # 1) Rod using tip vector (when sufficiently long)
        if rod_len >= 10:
            rod_tip_angle = (math.degrees(math.atan2(-dy_r, dx_r)) + 360.0) % 360.0
            # weight grows with length up to ~40px
            w_rt = max(0.0, min(1.0, (rod_len - 8.0) / 32.0)) * 1.2
            cand_angles.append(rod_tip_angle)
            cand_weights.append(w_rt)

        # 2) Rod using line fit direction (if available)
        if 'rod_line_angle' in locals() and rod_line_angle is not None:
            # couple with its quality metric
            w_rl = 0.9 * float(rod_line_conf)
            cand_angles.append(rod_line_angle)
            cand_weights.append(w_rl)

        # 3) Beak direction
        if beak_center is not None:
            dx_b = beak_center[0] - cx_local
            dy_b = beak_center[1] - cy_local
            if dx_b*dx_b + dy_b*dy_b > 4:
                beak_angle = (math.degrees(math.atan2(-dy_b, dx_b)) + 360.0) % 360.0
                # use modest weight; stronger when farther from center
                beak_dist = math.hypot(dx_b, dy_b)
                w_bk = max(0.0, min(1.0, (beak_dist - 2.0) / 18.0)) * 0.9
                cand_angles.append(beak_angle)
                cand_weights.append(w_bk)

        # 4) PCA body axis (choose flip closest to previous)
        axis_angle = (math.degrees(math.atan2(-axis[1], axis[0])) + 360.0) % 360.0
        cand_a = axis_angle
        cand_b = (axis_angle + 180.0) % 360.0
        if self._last_raw_angle is not None:
            da = abs(((cand_a - self._last_raw_angle + 540.0) % 360.0) - 180.0)
            db = abs(((cand_b - self._last_raw_angle + 540.0) % 360.0) - 180.0)
            axis_pick = cand_a if da <= db else cand_b
        else:
            axis_pick = cand_a
        # weight increases with eccentricity
        w_ax = max(0.0, min(1.0, (ecc - 1.4) / 2.6)) * 0.6
        cand_angles.append(axis_pick)
        cand_weights.append(w_ax)

        # if nothing valid, keep previous or axis_pick
        if len(cand_angles) == 0 or sum(cand_weights) < 1e-6:
            raw_angle = self._last_raw_angle if self._last_raw_angle is not None else axis_pick
        else:
            # circular weighted average
            ang_rad = np.deg2rad(np.array(cand_angles))
            w = np.asarray(cand_weights, dtype=np.float32)
            C = float(np.sum(w * np.cos(ang_rad)))
            S = float(np.sum(w * np.sin(ang_rad)))
            raw_angle = (math.degrees(math.atan2(S, C)) + 360.0) % 360.0

        # Low‑info frames: project previous angle forward instead of re‑deciding flip
        if low_info and self._last_raw_angle is not None:
            projected = (self._last_raw_angle + self._spin_rate_deg_s * dt) % 360.0
            raw_angle = projected

        # Update spin rate estimate when we have reasonable information
        if self._last_raw_angle is not None and (not low_info):
            diff = ((raw_angle - self._last_raw_angle + 540.0) % 360.0) - 180.0
            inst_rate = diff / max(dt, 1e-3)
            self._spin_rate_deg_s = 0.8 * self._spin_rate_deg_s + 0.2 * inst_rate

        # Store last raw
        self._last_raw_angle = raw_angle

        # circular smoothing (unchanged)
        if not hasattr(self, "_ang_ema"):
            self._ang_ema = raw_angle
        alpha = 0.25 if smooth_window > 1 else 1.0
        self._ang_ema = self._circular_ema(self._ang_ema, raw_angle, alpha)
        smoothed = float(self._ang_ema)
        self.smoothed_angle = smoothed

        if self.debug and (low_info or rod_len >= 12):
            dbg = {
                "raw": f"{raw_angle:.1f}",
                "sm": f"{smoothed:.1f}",
                "rod_len": f"{rod_len:.1f}",
                "beak": "Y" if beak_center else "N",
                "ecc": f"{ecc:.2f}",
                "low_info": low_info,
                "spin": f"{self._spin_rate_deg_s:.1f}"
            }
            print(f"[AngleDbg] {dbg}")
        
        if return_forward:
            return smoothed, self.facing_forward
        else:
            return smoothed
    
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