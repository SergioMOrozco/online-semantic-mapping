import torch
import clip
import cv2
import numpy as np
import os
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

img = cv2.imread("24.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# predictor.set_image(img)
# masks, _, _ = predictor.predict(["cannister","marker","cup"])
# masks, _, _ = predictor.predict("cannister")

# for mask in masks:
#     print(mask)

sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
_ = sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

file_list = os.listdir("../data/nerf/test/images")
out_dir = "../data/nerf/test/{}"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device="cuda")

for fl in file_list:
    img = cv2.imread(out_dir.format("images/"+fl))

    masks = mask_generator.generate(img)

    i = 0

    masked_image = np.zeros((img.shape[0],img.shape[1]))
    for mask in masks:

        # if i > 5:
        #     break
        # i +=1

        #print(mask)
        # if mask['stability_score'] < 0.99:
        #     continue

        res = cv2.bitwise_and(img,img,mask = mask['segmentation'].astype(np.uint8))


        # image = preprocess(Image.open("24.png")).unsqueeze(0).to(device)
        image = preprocess(Image.fromarray(res)).unsqueeze(0).to(device)
        text = clip.tokenize(["cannister", "marker", "cup", "water bottle", 'nerf gun', 'ball', 'table', 'floor', 'marker']).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print(["cannister", "marker", "cup", "water bottle", 'nerf gun', 'ball', 'table', 'floor', 'marker'])
        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        # cv2.imshow("img",res)
        # cv2.waitKey(0)

        max_id = np.argmax(probs)
        
        if np.max(probs) > 0.75:
            masked_image[mask['segmentation'] == True] = max_id + 1
            #masked_image[mask == False] = 255
        
    print(masked_image)
    # cv2.imshow("img",masked_image)
    # cv2.waitKey(0)
    #cv2.imwrite("masked.png",masked_image)

    cv2.imwrite(out_dir.format("segments/"+fl), masked_image)

        
