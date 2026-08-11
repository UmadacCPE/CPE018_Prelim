import cv2
import numpy

def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    """Apply edge detection and blend with the original image."""
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)

    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter that applies a convolution kernel to an image."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class BGRPortraCurveFilter(object):
    """A simple curve filter for portrait-like effect."""

    def __init__(self):
        # Example curve: brighten midtones
        self._lut = numpy.array([min(255, int(i * 1.1)) for i in range(256)], dtype=numpy.uint8)

    def apply(self, src, dst):
        # Apply LUT to each channel
        cv2.LUT(src, self._lut, dst)
