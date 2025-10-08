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
