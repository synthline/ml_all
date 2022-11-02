import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_images(**images):
    """Show multiple images using matplotlib."""
    # When a double-starred parameter is declared such as $**images$, then all
    # the keyword arguments from that point till the end are collected as a
    # dictionary called $'images'$.

    # Create a new matplotlib window.
    plt.figure()

    # Set the default colormap to gray and apply to current image if any.
    plt.gray()

    # Enumarate the ID, window name and images passed as parameter.
    for (pos, (name, image)) in enumerate(images.items()):
        # Show the image in a new subplot.
        plt.subplot(2, len(images) // 2, pos + 1)
        plt.title(name)
        plt.imshow(image)

    # Show the images.
    plt.show()


def shift_to_left(image, n):
    """Shift all pixel of the input image n column to the left."""
    result = image.copy()

    n = 2 * n + 1

    H = np.zeros((n, n))
    H[n // 2, n - 1] = 1.0

    result = cv2.filter2D(result, -1, H, borderType=cv2.BORDER_CONSTANT)

    return result


# Read image
image = cv2.imread("inputs/baboon.png", cv2.IMREAD_GRAYSCALE)


# <Exercise 8.3.2>

# <Exercise 8.3.3>
