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
    def __init__(self, player_images, fps=10):
        super().__init__()

        # Transparent overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Load player images into a list
        self.player_images = []
        for img in player_images:
            image = cv.imread(str(img), cv.IMREAD_GRAYSCALE)
            if image is not None:
                self.player_images.append(image)
        if len(self.player_images) < 8:
            raise ValueError("Not all player images could be loaded.")

        # ORB
        self.orb = cv.ORB_create(
            nfeatures=3000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=10
        )
        # Compute keypoints and descriptors for player images
        self.kps = []
        self.dess = []
        for img in self.player_images:
            kp, des = self.orb.detectAndCompute(img, None)
            self.kps.append(kp)
            self.dess.append(des)

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

        self.found = False
        self.angle = None
        if kp_frame is not None and des_frame is not None:
            best_matches = []
            for i, des in enumerate(self.dess):
                if des is not None and len(des) > 0:
                    matches = self.bf.match(des, des_frame)
                    matches = sorted(matches, key=lambda x: x.distance)
                    best_matches.append((i, matches))

            # Determine best matching player image
            best_img_index = None
            best_match_count = 0
            for i, matches in best_matches:
                if len(matches) > best_match_count and len(matches) > 10:  # Arbitrary threshold of 10 matches
                    best_match_count = len(matches)
                    best_img_index = i

            if best_img_index is not None:
                self.found = True
                # Calculate angle based on index (assuming order: front, SE, right, NE, back, NW, left, SW)
                self.angle = (best_img_index * 45) % 360
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

def debug_player_angle(player_images):
    app = QtWidgets.QApplication(sys.argv)
    overlay = PlayerOverlay(player_images, fps=5)
    # only overlay the top half of the screen
    screen = QtWidgets.QApplication.primaryScreen().geometry()
    overlay.setGeometry(0, 0, screen.width(), screen.height()//4)

    overlay.show()
    sys.exit(app.exec_())