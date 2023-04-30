import os
import cv2
import numpy as np
import pickle
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

# Load the pre-trained model
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
predictor = DefaultPredictor(cfg)

# Load the image
image = cv2.imread("./data/nerf/lab_med/images/0.png")
# Perform inference
outputs = predictor(image)

# Visualize the results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
outputs["panoptic_seg"][0].cpu()
out = v.draw_panoptic_seg(outputs["panoptic_seg"][0], outputs["panoptic_seg"][1])
#cv2.imshow("Semantic Segmentation", out.get_image()[:, :, ::-1])
#cv2.waitKey(0)

file_list = os.listdir("./data/nerf/lab_med/images")

#print(outputs['panoptic_seg'][0])
#cv2.imshow("Sem seg", outputs['panoptic_seg'][0])
#cv2.waitKey(0)
#plt.imshow(out.get_image()[:, :, ::-1])
#plt.show()

f_labels = {}

out_dir = "./data/nerf/lab_med/{}"

for fl in file_list:
    image = cv2.imread(out_dir.format("images/"+fl))
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    outputs["panoptic_seg"][0].cpu()
    f_labels[fl] = outputs["panoptic_seg"][0]
    out = v.draw_panoptic_seg(outputs["panoptic_seg"][0], outputs["panoptic_seg"][1])
    #cv2.imshow("Semantic Segmentation", out.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    cv2.imwrite(out_dir.format("segments/"+fl), out.get_image()[:, :, ::-1])

pickle.dump(f_labels, open(out_dir.format("semantic_labels.pkl"), 'wb'))
