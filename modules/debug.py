import sys
import cv2 as cv
import numpy as np
from mss import mss
import math
import pyautogui
from PyQt5 import QtWidgets, QtGui, QtCore
import vision


class FishOverlay(QtWidgets.QWidget):
    def __init__(self, x, y, w, h, threshold=15, fps=10, log_size=8):
        super().__init__()

        # Transparent overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # ROI (Region of Interest)
        self.roi = [x, y, w, h]
        self.threshold = threshold
        self.log = []           # text log of detections
        self.log_size = log_size
        self.last_detection = None  # (x, y) for visual marker

        # ORB (more sensitive)
        self.orb = cv.ORB_create(
            nfeatures=3000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=10
        )

        # Timer (for real-time updates)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(int(1000/fps))

        # Quit shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)

        # Close button
        self.close_btn = QtWidgets.QPushButton("X", self)
        self.close_btn.setFixedSize(40, 40)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(200, 0, 0, 180);
                color: white;
                border-radius: 20px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(255, 50, 50, 220);
            }
        """)
        self.close_btn.clicked.connect(self.close)
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.close_btn.move(screen.width() - 60, 20)

    def update_overlay(self):
        screenshot = pyautogui.screenshot()
        frame = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        detected = False
        detection_point = None

        if kp:
            x, y, w, h = self.roi
            inside = [(p.pt[0], p.pt[1]) for p in kp if x <= p.pt[0] <= x+w and y <= p.pt[1] <= y+h]

            if len(inside) >= self.threshold:
                detected = True
                # Average keypoint positions as detection center
                avg_x = int(np.mean([p[0] for p in inside]))
                avg_y = int(np.mean([p[1] for p in inside]))
                detection_point = (avg_x, avg_y + 30)  # draw slightly below ROI
                # Angle from center
                center_x = x + w // 2
                center_y = y + h // 2
                detection_angle = math.degrees(math.atan2(avg_y - center_y, avg_x - center_x))
                detection_angle = ((detection_angle + 270) % 360)/2

        # Update log
        if detected:
            self.last_detection = detection_point
            self.last_detection_angle = detection_angle
            msg = f"[{QtCore.QTime.currentTime().toString()}] Fish detected at {detection_point} with angle {detection_angle:.1f}°."
            self.log.append(msg)
        else:
            msg = f"[{QtCore.QTime.currentTime().toString()}] No fish."
            self.log.append(msg)

        # Limit log size
        if len(self.log) > self.log_size:
            self.log = self.log[-self.log_size:]

        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        x, y, w, h = self.roi

        # ROI rectangle
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 200), 2))
        painter.drawRect(x, y, w, h)

        # Draw detection marker
        if self.last_detection:
            fx, fy = self.last_detection
            angle = self.last_detection_angle
            painter.setBrush(QtGui.QColor(255, 0, 0, 220))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0, 200), 2))
            painter.drawEllipse(QtCore.QPoint(fx, fy), 8, 8)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220)))
            painter.setFont(QtGui.QFont("Consolas", 12))
            painter.drawText(fx + 15, fy + 5, f"({fx}, {fy}), {angle:.1f}°")

        # Draw text log bottom-left
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        painter.setFont(QtGui.QFont("Consolas", 12))
        y_offset = screen.height() - 20
        for entry in reversed(self.log):
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220)))
            painter.drawText(20, y_offset, entry)
            y_offset -= 20

class PlayerOverlay(QtWidgets.QWidget):
    def __init__(self, player_images, fps=10):
        super().__init__()

        # Transparent overlay setup
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Load reference images
        self.player_images = []
        for img in player_images:
            image = cv.imread(str(img), cv.IMREAD_GRAYSCALE)
            if image is not None:
                self.player_images.append(image)
        if len(self.player_images) == 0:
            raise ValueError("No player images could be loaded.")

        # AKAZE feature detector (rotation-aware)
        self.detector = cv.AKAZE_create(threshold=1e-4)

        # Compute keypoints/descriptors for templates
        self.kps, self.dess = [], []
        for img in self.player_images:
            kp, des = self.detector.detectAndCompute(img, None)
            self.kps.append(kp)
            self.dess.append(des)

        # Matcher
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # State variables
        self.angle = None
        self.angle_smooth = None
        self.smoothing_alpha = 0.2
        self.found = False
        self.show_debug = False

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(int(1000 / fps))

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("D"), self, activated=self.toggle_debug)

        # Close button
        self.close_btn = QtWidgets.QPushButton("X", self)
        self.close_btn.setFixedSize(40, 40)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(200, 0, 0, 180);
                color: white;
                border-radius: 20px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(255, 50, 50, 220);
            }
        """)
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.close_btn.move(screen.width() - 60, 20)

    # ------------------------------------------------------------

    def toggle_debug(self):
        self.show_debug = not self.show_debug
        if not self.show_debug:
            cv.destroyAllWindows()

    # ------------------------------------------------------------

    def update_overlay(self):
        screenshot = pyautogui.screenshot()
        frame = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Focus on ±400 px area around screen center
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - 400), max(0, cy - 400)
        x2, y2 = min(w, cx + 400), min(h, cy + 400)
        roi = gray[y1:y2, x1:x2]

        # Detect features in the ROI
        kp_frame, des_frame = self.detector.detectAndCompute(roi, None)
        self.found = False
        self.angle = None

        if kp_frame is None or des_frame is None or len(kp_frame) < 4:
            self.repaint()
            return

        # Try matching each template
        best_score = -np.inf
        best_angle = None
        best_vis = None

        for i, des in enumerate(self.dess):
            if des is None or len(des) == 0:
                continue

            matches = self.bf.match(des, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            good = [m for m in matches if m.distance < 70]
            if len(good) < 4:
                continue

            pts1 = np.float32([self.kps[i][m.queryIdx].pt for m in good])
            pts2 = np.float32([kp_frame[m.trainIdx].pt for m in good])

            # Estimate affine transform between template and ROI
            M, inliers = cv.estimateAffinePartial2D(pts1, pts2, method=cv.RANSAC)
            if M is None:
                continue

            angle_rad = np.arctan2(M[1, 0], M[0, 0])
            angle_deg = (np.degrees(angle_rad) + 360) % 360

            inlier_ratio = np.mean(inliers) if inliers is not None else 0
            score = inlier_ratio * len(good)

            if score > best_score:
                best_score = score
                best_angle = angle_deg
                if self.show_debug:
                    best_vis = cv.drawMatches(
                        self.player_images[i], self.kps[i],
                        roi, kp_frame,
                        good[:30], None,
                        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

        # Update final rotation estimate
        if best_angle is not None:
            self.found = True

            if self.angle_smooth is None:
                self.angle_smooth = best_angle
            else:
                diff = (best_angle - self.angle_smooth + 540) % 360 - 180
                self.angle_smooth = (self.angle_smooth + self.smoothing_alpha * diff) % 360

            self.angle = self.angle_smooth

            # Debug window
            if self.show_debug and best_vis is not None:
                cv.putText(best_vis, f"Angle: {self.angle:.1f}°",
                           (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.imshow("AKAZE Debug Matches", best_vis)
                cv.waitKey(1)

        self.repaint()

    # ------------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.found and self.angle is not None:
            text = f"Player Angle: {self.angle:.1f}°"
            color = QtGui.QColor(0, 255, 0, 220)
        else:
            text = "Player Not Found"
            color = QtGui.QColor(255, 0, 0, 220)
        painter.setFont(QtGui.QFont("Consolas", 16))
        painter.setPen(QtGui.QPen(color))
        painter.drawText(20, 40, text)       

def debug_player_angle(player_images):
    app = QtWidgets.QApplication(sys.argv)
    overlay = PlayerOverlay(player_images, fps=5)
    # only overlay the top half of the screen
    screen = QtWidgets.QApplication.primaryScreen().geometry()
    overlay.setGeometry(0, 0, screen.width(), screen.height()//4)

    overlay.show()
    sys.exit(app.exec_())

class MaskOverlay(QtWidgets.QWidget):
    def __init__(self, mode="white", fps=5):
        # ensure QApplication exists
        self._owns_app = False
        app = QtWidgets.QApplication.instance()
        if app is None:
            self.app = QtWidgets.QApplication([])
            self._owns_app = True
        else:
            self.app = app

        super().__init__()
        self.mode = mode
        self.count = 0
        self.mask = None

        screen_center = QtWidgets.QApplication.primaryScreen().geometry().center()
        self.roi = (screen_center.x() - 400, screen_center.y() - 400, 800, 800)

        # transparent always-on-top overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # configure window size to bottom third
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        w, h = screen.width(), screen.height()
        h_third = h // 3
        self.setGeometry(0, h - h_third, w, h_third)

        # Close button
        self.close_btn = QtWidgets.QPushButton("X", self)
        self.close_btn.setFixedSize(40, 40)
        self.close_btn.move(w - 60, 20)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(200,0,0,180);
                color: white;
                border-radius: 20px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(255,50,50,220);
            }
        """)
        self.close_btn.clicked.connect(self.close)

        # timer for live updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(int(1000 / fps))

        # keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("B"), self, activated=self.set_black_mode)
        QtWidgets.QShortcut(QtGui.QKeySequence("W"), self, activated=self.set_white_mode)

    def set_black_mode(self):
        self.mode = "black"

    def set_white_mode(self):
        self.mode = "white"

    def update_overlay(self):
        """Refresh mask with ROI ~800 px around the center and counter every frame."""
        self.mask = vision.screenshot_mask(self.mode, self.roi)
        self.count = cv.countNonZero(self.mask)
        self.update()  # triggers paintEvent

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setFont(QtGui.QFont("Consolas", 16))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220)))
        painter.drawText(20, self.height() - 40,
                         f"Mode: {self.mode.capitalize()} | Count: {self.count}")

    def run(self):
        """Show and start app loop if we own it."""
        self.show()
        if self._owns_app:
            self.app.exec_()

class RodAngleTracker(QtWidgets.QWidget):
    """Transparent overlay that detects and visualizes the penguin's fishing-rod angle."""

    def __init__(self, fps=10, alpha=0.25, roi_size=400):
        # QApplication setup
        app = QtWidgets.QApplication.instance()
        self._owns_app = False
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            self._owns_app = True
        self.app = app
        super().__init__()

        self.alpha = alpha
        self.smoothed_angle = None
        self.penguin_center = None
        self.rod_tip = None
        self.angle_text = "Angle: 0°"

        # --- Window setup (click-through translucent ROI overlay) ---
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        # Compute centered ROI
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        center = screen.center()
        x = center.x() - roi_size // 2
        y = center.y() - roi_size // 2
        w = h = roi_size
        self.roi = (x, y, w, h)

        # Only cover the ROI region, not full screen
        self.setGeometry(x, y, w, h)

        # Make the overlay mouse-transparent so clicks pass through
        self.setWindowFlag(QtCore.Qt.WindowTransparentForInput, True)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / fps))

        # Quit keys
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)

    def _tick(self):
        # capture ROI each frame
        with mss() as sct:
            x, y, w, h = self.roi
            grab = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h}))
            frame = cv.cvtColor(grab, cv.COLOR_BGRA2BGR)
        self._process(frame)
        self.update()

    def _process(self, frame):
        """Analyze frame to find penguin center, rod tip, facing direction, and rod angle."""
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # --- 1. Detect penguin (largest dark blob near center) ---
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv.inRange(hsv, lower_black, upper_black)
        mask_black = cv.morphologyEx(mask_black, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_black = cv.morphologyEx(mask_black, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv.findContours(mask_black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        penguin_contour = max(contours, key=cv.contourArea)
        M = cv.moments(penguin_contour)
        if M["m00"] == 0:
            return
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        self.penguin_center = (cx, cy)

        # --- 2. Detect fishing rod (light brown/yellow) ---
        lower_rod = np.array([10, 80, 80])
        upper_rod = np.array([35, 255, 255])
        mask_rod = cv.inRange(hsv, lower_rod, upper_rod)
        mask_rod = cv.morphologyEx(mask_rod, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_rod = cv.morphologyEx(mask_rod, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Keep only rod pixels above penguin center
        mask_height, mask_width = mask_rod.shape
        penguin_y = self.penguin_center[1]
        mask_rod[penguin_y:mask_height, :] = 0

        contours, _ = cv.findContours(mask_rod, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # Find the topmost rod point
        min_y = mask_height
        rod_tip = None
        for c in contours:
            y_min = c[:, :, 1].min()
            if y_min < min_y:
                min_y = y_min
                rod_tip = tuple(c[c[:, :, 1].argmin()][0])

        if rod_tip is None:
            return
        self.rod_tip = rod_tip

        # --- 3. Determine facing direction using white belly visibility ---
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask_white = cv.inRange(hsv, lower_white, upper_white)

        # Focus on lower central third of the ROI
        h, w = mask_white.shape
        roi_center = mask_white[h // 3 : h, w // 3 : 2 * w // 3]
        white_ratio = cv.countNonZero(roi_center) / roi_center.size

        self.facing_forward = white_ratio > 0.05  # threshold ~5% white pixels

        # --- 4. Compute rod angle relative to penguin center ---
        dx = self.rod_tip[0] - self.penguin_center[0]
        dy = self.rod_tip[1] - self.penguin_center[1]
        raw_angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

        # Mirror horizontally if penguin is facing away
        if not self.facing_forward:
            raw_angle = (540 - raw_angle) % 360  # mirror across vertical axis

        # --- 5. Apply smoothing to avoid jitter ---
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            delta = ((raw_angle - self.smoothed_angle + 540) % 360) - 180
            self.smoothed_angle = (self.smoothed_angle + self.alpha * delta) % 360

        # --- 6. Output text ---
        direction_label = "Forward" if self.facing_forward else "Away"
        self.angle_text = f"Angle: {self.smoothed_angle:.1f}° ({direction_label})"

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230)))
        painter.setFont(QtGui.QFont("Consolas", 16))
        painter.drawText(20, 40, self.angle_text)

        # draw debugging vector
        if self.penguin_center and self.rod_tip:
            px, py = self.penguin_center
            rx, ry = self.rod_tip
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 3))
            painter.drawLine(px, py, rx, ry)
            painter.setBrush(QtGui.QColor(0, 255, 0))
            painter.drawEllipse(px - 4, py - 4, 8, 8)
            painter.setBrush(QtGui.QColor(0, 0, 255))
            painter.drawEllipse(rx - 4, ry - 4, 8, 8)

    def run(self):
        self.show()
        if self._owns_app:
            sys.exit(self.app.exec_())

class FishingHUDOverlay(QtWidgets.QWidget):
    def __init__(self, fps=10):
        app = QtWidgets.QApplication.instance()
        self._owns_app = False
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            self._owns_app = True
        self.app = app
        super().__init__()

        # Vision backend
        self.vision = vision.VisionCortex(debug=False)

        # UI setup
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlag(QtCore.Qt.WindowTransparentForInput, True)
        self.setGeometry(0, 0, *self.vision.screen_size)

        # State
        self.fish_log = []
        self.log_size = 6
        self.last_fish = None
        self.splash_angle = None
        self.last_angle = 0
        self.direction_label = "?"
        self.motion_lines = []  # store motion contour lines

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(int(1000 / fps))

        # Exit keys
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)

    def update_overlay(self):
        # Run detectors
        splash_data = self.vision.update_splash_detector()   # (pt, angle) or None
        player_data = self.vision.update_player_detector()   # (angle, facing) or None

        # --- Track splash detections ---
        if splash_data:
            (pt, s_ang) = splash_data
            self.last_fish = pt
            msg = f"[{QtCore.QTime.currentTime().toString()}] Splash @ {pt}  ∠ {s_ang:.1f}°"
            self.fish_log.append(msg)
            self.splash_angle = s_ang
        if len(self.fish_log) > self.log_size:
            self.fish_log = self.fish_log[-self.log_size:]

        # --- Track player info ---
        if player_data:
            angle, facing = player_data
            self.last_angle = angle
            self.direction_label = "Forward" if facing else "Away"

        # --- Store current motion lines for overlay ---
        if hasattr(self.vision, "motion_contours") and self.vision.motion_contours:
            self.motion_lines = []
            for c in self.vision.motion_contours:
                pts = c.reshape(-1, 2)
                pts[:, 0] += self.vision.splashROI[0]
                pts[:, 1] += self.vision.splashROI[1]
                self.motion_lines.append(pts)
        else:
            self.motion_lines = []

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # --- 1. Draw Splash ROI ---
        x, y, w, h = self.vision.splashROI
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 128, 255, 180), 2))
        painter.drawRect(x, y, w, h)

        # --- 2. Draw motion contours ---
        if self.motion_lines:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 180), 2))
            for pts in self.motion_lines:
                for i in range(len(pts) - 1):
                    p1 = QtCore.QPoint(*pts[i])
                    p2 = QtCore.QPoint(*pts[i + 1])
                    painter.drawLine(p1, p2)
                # close contour loop
                if len(pts) > 2:
                    painter.drawLine(QtCore.QPoint(*pts[-1]), QtCore.QPoint(*pts[0]))

        # --- 3. Fish marker ---
        if self.last_fish:
            fx, fy = self.last_fish
            painter.setBrush(QtGui.QColor(255, 0, 0, 220))
            painter.drawEllipse(QtCore.QPoint(fx, fy), 8, 8)
            if self.splash_angle is not None:
                painter.setFont(QtGui.QFont("Consolas", 12))
                painter.setPen(QtGui.QColor(255, 255, 255, 230))
                painter.drawText(fx + 14, fy, f"{self.splash_angle:.1f}°")

        # --- 4. Player info ---
        painter.setFont(QtGui.QFont("Consolas", 16))
        painter.setPen(QtGui.QColor(255, 255, 255, 230))
        painter.drawText(20, 40, f"Rod: {self.last_angle:.1f}° ({self.direction_label})")

        # --- 5. Rod visualization ---
        if self.vision.penguin_center and self.vision.rod_tip:
            px, py = self.vision.penguin_center
            rx, ry = self.vision.rod_tip
            x, y, w, h = self.vision.penguinROI
            px += x
            py += y
            rx += x
            ry += y
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 3))
            painter.drawLine(px, py, rx, ry)
            painter.setBrush(QtGui.QColor(0, 255, 0))
            painter.drawEllipse(px - 4, py - 4, 8, 8)
            painter.setBrush(QtGui.QColor(255, 255, 0))
            painter.drawEllipse(rx - 4, ry - 4, 8, 8)

        # --- 6. Log ---
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        painter.setFont(QtGui.QFont("Consolas", 12))
        y_offset = screen.height() - 20
        for entry in reversed(self.fish_log):
            painter.drawText(20, y_offset, entry)
            y_offset -= 20

    def run(self):
        self.show()
        if self._owns_app:
            sys.exit(self.app.exec_())