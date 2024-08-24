from banner_placement import BannerPlacer
from facade_processing import FacadeProcessor
import time
import os


def get_sources_path():
    """
    Returns the paths for all the sources that are in use during the program
    """
    sources_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sources')
    image_path = os.path.join(sources_path, 'image.jpg')
    scanned_data_path = os.path.join(sources_path, '3d_scan.txt')
    banner_coor_path = os.path.join(sources_path, 'coordinated.png')
    banner_img_path = os.path.join(sources_path, 'banner.jpg')
    projMat_path = os.path.join(sources_path, 'projMat.txt')
    return image_path, scanned_data_path, banner_coor_path, banner_img_path, projMat_path


def main():
    start = time.time()
    image_path, scanned_data_path, banner_coordinates_path, banner_image_path, camera_matrix_path = get_sources_path()
    # Process banner placement
    banner_placer = BannerPlacer(camera_matrix_path, banner_image_path, banner_coordinates_path, image_path)
    banner_placer.run()
    # Process facade cropping
    facade_processor = FacadeProcessor(scanned_data_path, banner_placer)
    facade_processor.run()
    print("Total runtime: ", time.time() - start)
    facade_processor.display_result()


if __name__ == "__main__":
    main()
