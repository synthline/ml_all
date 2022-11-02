import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt


def main():

    data, video, image_map = load_data()

    Hgm = load_or_create_affine(video, image_map)
    i = 0

    while True:
        ret, image = video.read()
        if not ret:
            break

        draw_rectangles(image, data, i)

        # Exercise 2.3

        i += 1

        cv2.imshow("map", image_map)
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    video.release()


def load_or_create_affine(video, image_map):
    """Loads transform from file if it exists, otherwise creates a new one."""
    output = "transform.npy"

    # Check if saved file exists
    if os.path.isfile(output):
        Hgm = np.load(output)
    else:
        ret, image_ground = video.read()
        if not ret:
            raise IOError("Could not read frame from Video.")

        # Exercise
        points_source = Counter(image_ground, "Ground points").get_points(3)
        points_destination = Counter(image_map, "Map points").get_points(3)
        # Replace this with your solution
        Hgm = None

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return Hgm


# Exercise
def apply_affine(t, point):
    """Apply affine transformation h to point."""
    return None # Replace


class Counter:
    """UI element that allows you to mark points in an image."""

    def __init__(self, img, title) -> None:
        """Set up the window and callback.

        Args:
            img: image to show.
            title: window title.
        """
        self.img = img.copy()
        self.window = title
        self.down = False
        self.points = []

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.callback)
        self.update()

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.down:
                self.down = True
                self.points.append([x, y])
                cv2.drawMarker(self.img, (x, y), 0, cv2.MARKER_CROSS, 10, 2)
        if event == cv2.EVENT_LBUTTONUP:
            self.down = False

        self.update()

    def update(self):
        cv2.imshow(self.window, self.img)

    def get_points(self, n):
        """Get n points from image.

        Can be called multiple times. Doesn't reset points between calls.

        Args:
            n: number of points.

        Returns:
            np.ndarray: fetched points in x, y coordinates as float.
        """
        while len(self.points) < n:
            cv2.waitKey(1)
        return np.array(self.points, dtype=np.float32)

    def __del__(self):
        cv2.destroyWindow(self.window)


def to_homogeneous(points):
    if len(points.shape) == 1:
        points = points.reshape((*points.shape, 1))
    return np.vstack((points, np.ones((1, points.shape[1]))))


def to_euclidean(points):
    return points[:2] / points[2]


def load_data():
    """Loads the tracking data, the input video, and the map image."""
    filename = "inputs/trackingdata.dat"
    data = np.loadtxt(filename)
    data = {"body": data[:, :4], "legs": data[:, 4:8], "all": data[:, 8:]}

    videofile = "inputs/ITUStudent.mov"
    video = cv2.VideoCapture(videofile)

    imagename = "inputs/ITUMap.png"
    image_map = cv2.imread(imagename)

    return data, video, image_map


def draw_part(image, part, color, i):
    """Draw rectangle of specific body part at time i."""
    cv2.rectangle(
        image,
        tuple(part[i, 0:2].astype(int)),
        tuple(part[i, 2:4].astype(int)),
        color,
        thickness=1,
    )


def draw_rectangles(image, data, i):
    """Draw all body parts at time i."""
    draw_part(image, data["body"], (0, 0, 255), i)
    draw_part(image, data["legs"], (255, 0, 0), i)
    draw_part(image, data["all"], (0, 255, 0), i)


def get_center(part, i):
    """Returns center of body part in homogeneous coordinates.

    Parameters: part refers to a Nx4 array containing rectangle points for a specific
    body part. i refers to the frame index to fetch.
    """
    x = int((part[i, 0] + part[i, 2]) / 2)
    y = int((part[i, 1] + part[i, 3]) / 2)

    return to_homogeneous(np.array([x, y]))


if __name__ == "__main__":
    main()