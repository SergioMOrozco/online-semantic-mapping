class CameraIntrinsics:
    def __init__(self, camera_angle_x, camera_angle_y, fl_x, fl_y,
                 k1, k2, p1, p2, cx, cy, w, h, aabb_scale):
        #self.camera_angle_x = camera_angle_x
        #self.camera_angle_y = camera_angle_y
        #self.fl_x = fl_x
        #self.fl_y = fl_y
        #self.k1 = k1
        #self.k2 = k2
        #self.p1 = p1
        #self.p2 = p2
        #self.cx = cx
        #self.cy = cy
        #self.w = w
        #self.h = h
        #self.aabb_scale = aabb_scale

        self.camera_angle_x= 1.0444974381925791
        self.camera_angle_y= 0.8125453711687044
        self.fl_x= 555.9883821369561
        self.fl_y= 557.8710673609644
        self.k1= 0.11665685258396481
        self.k2= -0.21312256474967875
        self.k3= 0
        self.k4= 0
        self.p1= -0.006740129216126598
        self.p2= -0.0031901714634920545
        self.is_fisheye= False
        self.cx= 322.19647507987685
        self.cy= 230.93252628078122
        self.w= 640.0
        self.h= 480.0
        self.aabb_scale= 32
