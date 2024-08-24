import open3d as o3d
import numpy as np
import itertools
import cv2

class FacadeProcessor:
    def __init__(self, scanned_data_path, banner_placer):
        pcd, _ = self.load_points_data(scanned_data_path).remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.point_cloud = pcd.voxel_down_sample(voxel_size=0.04)  # uniformly downsampled point cloud
        self.banner_placer = banner_placer
        self.right_facade = None
        self.cropped_image = None

    def run(self):
        self.find_right_facade()
        self.compute_and_crop_right_facade_bb()

    def find_right_facade(self):
        """
        Separates the points cloud into 2 planes using RANSAC.
        Chooses the relevant plane by using the banner coordinates (that are places in the right facade).
        Filters outliers using remove_statistical_outlier.
        :return: right_facade_pcd: pointCloud object of the right facade
        """
        m, c, p = 3, 0.99, 0.6
        t = int(np.log(1 - c) / np.log(1 - (1 - p) ** m))
        # Apply RANSAC to detect the planes
        _, inliers = self.point_cloud.segment_plane(distance_threshold=0.06, ransac_n=m, num_iterations=t)

        # Extract inlier points corresponding to the plane
        inlier_cloud = self.point_cloud.select_by_index(inliers)
        remaining_cloud = self.point_cloud.select_by_index(inliers, invert=True)
        # Calculates the distance of each point (from banner coordinates) to the planner points and keep the planar with
        # minimum distance (I know the banner is on the right facade)
        distances = [
            sum((np.linalg.norm(p - np.asarray(plane_pcd.points), axis=1).mean()) for p in self.banner_placer.get_3d_coordinates())
            for plane_pcd in [inlier_cloud, remaining_cloud]]
        right_facade_pcd = [inlier_cloud, remaining_cloud][np.argmin(distances)]
        # remove outliers
        right_facade_pcd_, ind = right_facade_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        # display_inlier_outlier(right_facade_pcd, ind)
        self.right_facade = right_facade_pcd_

    def compute_and_crop_right_facade_bb(self):
        """
        Computes the bounding box of the right facade in 3d world and than project it into the 2d world using camera matrix
        """
        camera_matrix = self.banner_placer.camera_matrix
        # Define the 8 corners (3D bounding box) of the right facade bounding box
        aabb = self.right_facade.get_axis_aligned_bounding_box()
        min_bound = np.array(aabb.min_bound)
        max_bound = np.array(aabb.max_bound)
        bounding_box_corners_3d = np.array(list(itertools.product(*zip(min_bound, max_bound))))
        # Convert to homogeneous coordinates
        bounding_box_corners_homogeneous = np.hstack(
            (bounding_box_corners_3d, np.ones((bounding_box_corners_3d.shape[0], 1))))
        # Project the 3D points onto the 2D image plane
        image_bounding_box_corners_homogeneous = camera_matrix @ bounding_box_corners_homogeneous.T  # 3 x 8
        bounding_box_corners_2d = np.array((image_bounding_box_corners_homogeneous[:2, :] /
                                            image_bounding_box_corners_homogeneous[2, :]).T, dtype=np.int32)  # 8 x 2
        # Find bounding box from the projected 2D points
        x, y, w, h = cv2.boundingRect(bounding_box_corners_2d)
        self.cropped_image = self.banner_placer.get_result_image()[:, x:x + w]

    def display_result(self):
        cv2.imshow("Cropped Image", self.cropped_image)
        cv2.waitKey(0)

    @staticmethod
    def load_points_data(points_path: str):
        """
        Opens the scan 3d points and creates PointCloud object.
        :param points_path: path to 3d_scan text file.
        :return: PointCloud object with the given scan points
        """
        data = np.loadtxt(points_path, delimiter=',')
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0  # Normalize RGB values to [0, 1] range (for o3d.visualization)
        normals = data[:, 6:9]
        # Creates an Open3D PointCloud object and initializes it with the given data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd
