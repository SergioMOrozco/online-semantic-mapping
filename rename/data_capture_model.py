from __future__ import unicode_literals
import json
import io
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

        camera_intrinsics = self.__camera_intrinsics.__dict__
        frames = {"frames": [frame.__dict__ for frame in self.frames]}
        
        j = self.__merge_two_dicts(camera_intrinsics, frames)

        j = json.dumps(j, indent=4)

        with open(self.file_name, 'w') as f:
            print >> f, j

    def __merge_two_dicts(self, x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z
