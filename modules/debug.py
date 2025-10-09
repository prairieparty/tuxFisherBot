import sys
import cv2 as cv
import numpy as np
import vision
from mss import mss
import pyautogui
from PyQt5 import QtWidgets, QtGui, QtCore

class FishOverlay(QtWidgets.QWidget):
    def __init__(self, x, y, w, h, threshold=15, fps=10, log_size=8):
        super().__init__()

        # Transparent overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # ROI
        self.roi = [x, y, w, h]
        self.threshold = threshold
        self.log = []           # text log of detections
        self.log_size = log_size

        # ORB
        self.orb = cv.ORB_create(
            nfeatures=3000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=10
        )

        # Timer
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
        if kp:
            x, y, w, h = self.roi
            inside = [p for p in kp if x <= p.pt[0] <= x+w and y <= p.pt[1] <= y+h]
            if len(inside) >= self.threshold:
                detected = True

        # Update log
        if detected:
            msg = f"[{QtCore.QTime.currentTime().toString()}] Fish detected in ROI!"
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

        # Draw log text bottom-left
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        painter.setFont(QtGui.QFont("Consolas", 12))
        y_offset = screen.height() - 20
        for entry in reversed(self.log):
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220)))
            painter.drawText(20, y_offset, entry)
            y_offset -= 20

def debug_orb_keypoints():
    app = QtWidgets.QApplication(sys.argv)
    overlay = FishOverlay(x=0, y=300, w=2600, h=300, threshold=15, fps=5, log_size=10)
    overlay.showFullScreen()
    sys.exit(app.exec_())

def getBlackMasking(img=vision.screenshot_roi()):
    # given screenshot image, return black areas masked as white, rest black
    frame = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    #convert to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30]) # value up to 30 for dark/black areas
    # create mask
    mask = cv.inRange(hsv, lower_black, upper_black)
    return mask

def getWhiteMasking(img=vision.screenshot_roi()):
    # given screenshot image, return white areas masked as white, rest black
    frame = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    #convert to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range for white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    # create mask
    mask = cv.inRange(hsv, lower_white, upper_white)
    return mask

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

def saveBlackAndWhiteMasks():
    black_mask = getBlackMasking()
    white_mask = getWhiteMasking()
    cv.imwrite("black_mask.png", black_mask)
    cv.imwrite("white_mask.png", white_mask)
    print("Black and white masks saved as black_mask.png and white_mask.png")

def debug_player_angle(player_images):
    app = QtWidgets.QApplication(sys.argv)
    overlay = PlayerOverlay(player_images, fps=5)
    # only overlay the top half of the screen
    screen = QtWidgets.QApplication.primaryScreen().geometry()
    overlay.setGeometry(0, 0, screen.width(), screen.height()//4)

    overlay.show()
    sys.exit(app.exec_())