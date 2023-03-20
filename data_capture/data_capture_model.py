import json
from bosdyn.client import math_helpers
from frame import Frame
from camera_intrinsics import CameraIntrinsics

class DataCaptureModel:
    def __init__(self, file_name):
        self.file_name = file_name
        self.frames = []
        self.__camera_intrinsics = CameraIntrinsics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def set_camera_instrinsics(self, camera_angle_x, camera_angle_y, fl_x, fl_y,
                 k1, k2, p1, p2, cx, cy, w, h, aabb_scale):

        self.__camera_intrinsics = CameraIntrinsics(camera_angle_x, camera_angle_y, fl_x,
                                                    fl_y, k1, k2, p1, p2, cx, cy, w, h, aabb_scale)

    def add_frame(self, image_path, transformation_matrix):

        # Assume 100 sharpness for now
        frame = Frame(image_path, 100, transformation_matrix)
        self.frames.append(frame)

    def clear_frames(self):
        self.matrices = {}

    def write_to_file(self):
        with open('transformation_matrix.json', 'w', encoding='utf-8') as f:
            camera_intrinsics = self.__camera_intrinsics.__dict__
            frames = {"frames": [frame.__dict__ for frame in self.frames]}

            j = self.__merge_two_dicts(camera_intrinsics, frames)

            json.dump(j, f, ensure_ascii=False, indent=4)

    def __merge_two_dicts(self, x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z

if __name__ == "__main__":
    data_capture = DataCaptureModel("test.json")

    test_1 = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat(0, 2, 0, 0))
    test_2 = math_helpers.SE3Pose(1, 2, 3, math_helpers.Quat(0, 1, 0, 0))
    data_capture.set_camera_instrinsics(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

    data_capture.add_frame("test_1.png", test_1.to_matrix())
    data_capture.add_frame("test_2.png", test_2.to_matrix())

    data_capture.write_to_file()

