import re
import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class BannerPlacer:
    def __init__(self, camera_matrix_path, banner_image_path, banner_coordinates_path, original_image_path):
        self.camera_matrix = np.loadtxt(camera_matrix_path)
        self.banner_image = cv2.imread(banner_image_path)
        self.banner_coordinates_image = cv2.imread(banner_coordinates_path, 0)
        self.original_image = cv2.imread(original_image_path)
        self.banner_3d_coordinates = []
        self.result_image = None

    def run(self):
        self.transform_and_place()

    def extract_coordinates_from_image(self):
        text = pytesseract.image_to_string(self.banner_coordinates_image)
        return self.parse_coordinates(text)

    def transform_and_place(self):
        """
        transforming the banner image so it fit to the given coordinates using getPerspectiveTransform
        and place it in the correct place in the original image using a mask.
        :returns
        numpy array: The original image with the banner placed on it.
        """
        # Gets banner's coordinates and transform it into 2D by the projMat
        self.banner_3d_coordinates = self.extract_coordinates_from_image()
        banner_coordinates_homogeneous = np.hstack((np.vstack(self.banner_3d_coordinates), np.ones((4, 1))))
        image_banner_coordinates_homogeneous = self.camera_matrix @ banner_coordinates_homogeneous.T
        banner_coordinates_2d = np.array((image_banner_coordinates_homogeneous[:2, :] /
                                          image_banner_coordinates_homogeneous[2, :]).T, dtype=np.int32)

        # Define the four corners of the banner image (source points)
        banner_height, banner_width = self.banner_image.shape[:2]
        src_points = np.float32([[0, 0], [banner_width, 0], [0, banner_height], [banner_width, banner_height]])
        # Ensure dst_points is a numpy array of type float32 (for using in cv2.getPerspectiveTransform)
        dst_points = np.float32(banner_coordinates_2d)

        # Get the perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        # Warp the banner image to fit the destination coordinates
        h, w = self.original_image.shape[:2]
        warped_banner = cv2.warpPerspective(self.banner_image, transform_matrix, (w, h))

        # Create a mask from the warped banner image
        banner_gray = cv2.cvtColor(warped_banner, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(banner_gray, 1, 255, cv2.THRESH_BINARY)
        # Make sure the mask has no holes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.bool_)

        self.result_image = self.original_image.copy()
        self.result_image[mask] = warped_banner[mask]

    def get_3d_coordinates(self):
        return self.banner_3d_coordinates

    def get_result_image(self):
        return self.result_image

    def parse_coordinates(self, text):
        """
        Splits the coordinates by using expression that find them in the text.
        :param text: The output of the OCR
        :return: tuple of sorted coordinates
        """
        # Regular expression to match coordinates
        pattern = r'\((-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
        matches = re.findall(pattern, text)
        coordinates = [(float(x), float(y), float(z)) for x, y, z in matches]
        assert len(coordinates) == 4, "OCR didn't work as accepted, didn't find in the text 4 coordinates"
        return self.sort_coordinates(coordinates)

    @staticmethod
    def sort_coordinates(coordinates):
        """
        Sorts the given coordinates into top_left, top_right, bottom_left, bottom_right
        :param coordinates: tuple of coordinates
        :return: tuple of sorted coordinates
        """
        coords = coordinates.copy()
        # Identify the top-left, top-right, bottom-left, and bottom-right points
        top_left = min(coords, key=lambda p: (p[0] + p[1]))
        bottom_right = max(coords, key=lambda p: (p[0] + p[1]))
        coords.remove(top_left)
        coords.remove(bottom_right)
        top_right = max(coords, key=lambda p: p[0])
        bottom_left = min(coords, key=lambda p: p[0])
        return top_left, top_right, bottom_left, bottom_right


