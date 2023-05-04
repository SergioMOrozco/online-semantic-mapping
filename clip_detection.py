import torch
import clip
import cv2
import numpy as np
import os
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
torch.cuda.empty_cache()

#img = cv2.imread("24.png")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

file_list = os.listdir("images")
out_dir = "{}"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device="cuda")


random_colors = []
random_colors.append([256,256,256])
random_colors.append([256,0,0]) # blue
random_colors.append([0,255,0]) # green
random_colors.append([0,0,255]) # red
random_colors.append([0,255,255]) # yellow
random_colors.append([240,32,160]) # purple
random_colors.append([0,75,150]) # brown
random_colors.append([203,192,255]) # pink
random_colors.append([100,100,100]) # grey
random_colors.append([31,95,255]) # orange
random_colors.append([128,128,0])

for fl in file_list:
    print(fl)
    img = cv2.imread(out_dir.format("images/"+fl))

    masks = mask_generator.generate(img)

    i = 0

    masked_image = np.zeros((img.shape[0],img.shape[1]))
    masked_image_visual = np.zeros((img.shape[0],img.shape[1],3))
    for mask in masks:

        if i > 10:
            break
        i +=1

        #print(mask)
        # if mask['stability_score'] < 0.99:
        #     continue

        res = cv2.bitwise_and(img,img,mask = mask['segmentation'].astype(np.uint8))


        # image = preprocess(Image.open("24.png")).unsqueeze(0).to(device)
        image = preprocess(Image.fromarray(res)).unsqueeze(0).to(device)
        text = clip.tokenize(["scissors", "toy robot", "water bottle", "book", 'table', 'cup', 'floor', 'marker']).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        #print(["scissors", "toy robot", "water bottle", "book", 'table', 'cup', 'floor', 'marker','block'])
        #print(["cannister", "marker", "cup", "water bottle", 'nerf gun', 'ball', 'table', 'floor', 'marker'])
        #print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        # cv2.imshow("img",res)
        # cv2.waitKey(0)

        max_id = np.argmax(probs)
        
        if np.max(probs) > 0.75:
            masked_image[mask['segmentation'] == True] = max_id + 1
            masked_image_visual[mask['segmentation'] == True] = random_colors[max_id + 1]
        
    #print(masked_image)
    # cv2.imshow("img",masked_image)
    # cv2.waitKey(0)
    #cv2.imwrite("masked.png",masked_image)

    cv2.imwrite(out_dir.format("segments/"+fl), masked_image)
    cv2.imwrite(out_dir.format("segments_visual/"+fl), masked_image_visual)

        
