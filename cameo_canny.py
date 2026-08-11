import cv2
from managers import WindowManager, CaptureManager

class CameoCanny(object):
    def __init__(self):
        self._windowManager = WindowManager('Canny Detection', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """Run the main loop with Canny edge detection."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                # Replace frame with edges (convert back to BGR for consistency)
                frame[:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        if keycode == 32:  # space
            self._captureManager.writeImage('canny_screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('canny_screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    CameoCanny().run()
