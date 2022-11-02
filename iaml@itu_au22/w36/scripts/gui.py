import click
import cv2 as cv


class SimpleApplication:

    # Constructor
    def __init__(self, input_path, output_path):
        self.base_img = ...
        self.cropped = self.base_img
        self.img_height, self.img_width, _ = self.base_img.shape
        self.crop_x = self.img_width
        self.output_path = output_path
        self.blur = 1
        self.key = None


        # Application main loop
        while self.key != ord("q"):
            self.update()

        # Window cleanup
        cv.destroyAllWindows()
        cv.waitKey(1)

    def update(self):
        ...
        # Get keypress and call handler
        self.key = cv.waitKey(1)
        self.handle_key_events()

    def handle_key_events(self):
        ...

    def handle_width_update(self, value):
        ...



@click.command()
@click.argument("Input")
@click.argument("Output")
def run(input, output):
    """Sample CV application"""
    SimpleApplication(input, output)


if __name__ == "__main__":
    run()