class Frame:
    def __init__(self,file_path,sharpness,transform_matrix):
        self.file_path = file_path
        self.sharpness = sharpness
        self.transform_matrix = transform_matrix.tolist()
