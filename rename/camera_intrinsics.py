class CameraIntrinsics:
    def __init__(self, camera_angle_x, camera_angle_y, fl_x, fl_y,
                 k1, k2, p1, p2, cx, cy, w, h, aabb_scale):
        self.camera_angle_x = camera_angle_x
        self.camera_angle_y = camera_angle_y
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.aabb_scale = aabb_scale
