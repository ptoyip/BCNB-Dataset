import os
import json
from glob import iglob

import cv2
import matplotlib.pyplot as plt
import numpy as np

#FIXME: #! Please increase opencv's validateInputImageSize

if __name__ == "__main__":
    wsi_img_folder = sorted(iglob(r"BCNB Dataset/WSIs/*.jpg"))
    wsi_json_folder = sorted(iglob(r"BCNB Dataset/WSIs/*.json"))
    if not os.path.exists(r"BCNB Dataset/wsi_patches"):
        os.mkdir(r"BCNB Dataset/wsi_patches")

    for idx, json_file in enumerate(wsi_json_folder):
        wsi = cv2.imread(wsi_img_folder[idx])
        
        with open(json_file) as f:
            if not os.path.exists(r"BCNB Dataset/wsi_patches/" + str(idx)):
                os.mkdir(r"BCNB Dataset/wsi_patches/" + str(idx))
            json_data = json.load(f)
            # * One WSI Image
            bboxes = []
            vertices = []
            for annotation in json_data["positive"]:
                vertices.append(annotation["vertices"])
                bboxes.append(
                    [
                        int(min(i[0] for i in annotation["vertices"])),
                        int(max(i[0] for i in annotation["vertices"])),
                        int(min(i[1] for i in annotation["vertices"])),
                        int(max(i[1] for i in annotation["vertices"])),
                    ]
                )
            
            # * bbox now contain all bbox within a WSI
            for bbox_idx, bbox in enumerate(bboxes):
                min_x, max_x, min_y, max_y = bbox
                x, y, width, height = min_x, min_y, max_x - min_x, max_y - min_y

                bbox_pt = np.array([vertices[bbox_idx]]) - np.array([min_x, min_y])
                img_bbox = wsi[min_y:max_y, min_x:max_x]
                
                black = np.zeros(img_bbox.shape[:2]).astype(img_bbox.dtype)
                blur_mask = cv2.fillPoly(black, [bbox_pt], color=255)
                img_bbox_mask = np.absolute(255 - blur_mask)

                # tmp_result = cv2.bitwise_and(img_bbox,block_inv)
                blur = cv2.GaussianBlur(img_bbox, (33, 33), 33)
                result = np.copy(img_bbox)

                result[np.where(img_bbox_mask == 255)] = blur[
                    np.where(blur_mask == 0)
                ]
                # cv2.imwrite(
                #     r"BCNB Dataset/wsi_patches/"
                #     + str(idx)
                #     + "/"
                #     + str(bbox_idx)
                #     + ".jpg",
                #     blur_mask,
                # )
                cv2.imwrite(
                    r"BCNB Dataset/wsi_patches/"
                    + str(idx)
                    + "/"
                    + str(bbox_idx)
                    + "d-b.jpg", # detailed-blured
                    result,
                )