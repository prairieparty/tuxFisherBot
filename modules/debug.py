import sys
import cv2 as cv
import numpy as np
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

class PlayerOverlay(QtWidgets.QWidget):
    def __init__(self, player_front_path, player_back_path, player_left_path, player_right_path, fps=10):
        super().__init__()

        # Transparent overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Load player images
        self.player_front = cv.imread(player_front_path, cv.IMREAD_GRAYSCALE)
        self.player_back = cv.imread(player_back_path, cv.IMREAD_GRAYSCALE)
        self.player_left = cv.imread(player_left_path, cv.IMREAD_GRAYSCALE)
        self.player_right = cv.imread(player_right_path, cv.IMREAD_GRAYSCALE)
        if self.player_front is None or self.player_back is None or self.player_left is None or self.player_right is None:
            raise ValueError("Player images not found or could not be loaded.")

        # ORB
        self.orb = cv.ORB_create(
            nfeatures=3000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=10
        )
        self.kp_front, self.des_front = self.orb.detectAndCompute(self.player_front, None)
        self.kp_back, self.des_back = self.orb.detectAndCompute(self.player_back, None)
        self.kp_left, self.des_left = self.orb.detectAndCompute(self.player_left, None)
        self.kp_right, self.des_right = self.orb.detectAndCompute(self.player_right, None)

        # BFMatcher
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

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
        self.angle = None
        self.found = False
    def update_overlay(self):
        screenshot = pyautogui.screenshot()
        frame = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)

        matches_front = []
        matches_back = []
        matches_left = []
        matches_right = []
        
        if des_frame is not None:
            matches_front = self.bf.match(self.des_front, des_frame)
            matches_back = self.bf.match(self.des_back, des_frame)
            matches_left = self.bf.match(self.des_left, des_frame)
            matches_right = self.bf.match(self.des_right, des_frame)

        # Sort matches by distance
        matches_front = sorted(matches_front, key=lambda x: x.distance)
        matches_back = sorted(matches_back, key=lambda x: x.distance)
        matches_left = sorted(matches_left, key=lambda x: x.distance)
        matches_right = sorted(matches_right, key=lambda x: x.distance)

        # Determine which has more good matches
        good_matches_front = [m for m in matches_front if m.distance < 50]
        good_matches_back = [m for m in matches_back if m.distance < 50]
        good_matches_left = [m for m in matches_left if m.distance < 50]
        good_matches_right = [m for m in matches_right if m.distance < 50]

        match_counts = {
            'front': len(good_matches_front),
            'back': len(good_matches_back),
            'left': len(good_matches_left),
            'right': len(good_matches_right)
        }
        best_view = max(match_counts, key=match_counts.get)
        best_count = match_counts[best_view]
        if best_count >= 10:  # Arbitrary threshold for detection
            self.found = True
            if best_view == 'front':
                self.angle = 0.0
            elif best_view == 'right':
                self.angle = 90.0
            elif best_view == 'back':
                self.angle = 180.0
            elif best_view == 'left':
                self.angle = 270.0
        else:
            self.found = False
            self.angle = None
        self.repaint()
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        screen = QtWidgets.QApplication.primaryScreen().geometry()

        if self.found and self.angle is not None:
            text = f"Player Angle: {self.angle:.2f}°"
            color = QtGui.QColor(0, 255, 0, 220)
        else:
            text = "Player Not Found"
            color = QtGui.QColor(255, 0, 0, 220)

        # Draw text top-left
        painter.setFont(QtGui.QFont("Consolas", 16))
        painter.setPen(QtGui.QPen(color))
        painter.drawText(20, 40, text)

def debug_player_angle(player_front_path, player_back_path, player_left_path, player_right_path):
    app = QtWidgets.QApplication(sys.argv)
    overlay = PlayerOverlay(player_front_path, player_back_path, player_left_path, player_right_path, fps=5)
    overlay.showFullScreen()
    sys.exit(app.exec_())