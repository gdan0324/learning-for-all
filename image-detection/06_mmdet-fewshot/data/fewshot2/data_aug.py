import os
import json
import numpy as np
import cv2


def mosaic(img_path_list):
    # 4张图合并，中间留10像素空隙
    imgs = []
    h1 = 0
    w1 = 0
    hs = []
    ws = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        imgs.append(img)
        h1 = max(img.shape[0], h1)
        w1 = max(img.shape[1], w1)
        hs.append(img.shape[0])
        ws.append(img.shape[1])
    new_img = np.zeros((2 * h1 + 10, 2 * w1 + 10, 3))
    new_img[:hs[0], :ws[0], :] += imgs[0]
    new_img[:hs[1], w1 + 10:w1 + 10 + ws[1], :] += imgs[1]
    new_img[h1 + 10:h1 + 10 + hs[2], :ws[2], :] += imgs[2]
    new_img[h1 + 10:h1 + 10 + hs[3], w1 + 10:w1 + 10 + ws[3], :] += imgs[3]
    return new_img, h1, w1


def file_name_to_image_id(file_name, json_file):
    for image in json_file["images"]:
        if image["file_name"] == file_name:
            return image["id"]
    raise KeyError("没找到%s的image_id" % file_name)


def image_id_to_bbox(image_id, json_file):
    bboxes = []
    for anno in json_file["annotations"]:
        if anno["image_id"] == image_id:
            bboxes.append(anno["bbox"])
    return bboxes


if __name__ == '__main__':
    json_file = json.load(open("train_annotation.json", 'r'))
    image_id = len(json_file["images"])
    box_id = len(json_file["annotations"])
    # 所以图片
    all_imgs = os.listdir("train/JPEGImages")
    # 有标签的图片
    anno_imgs = [each["file_name"] for each in json_file["images"]]
    # 无标签的图片
    normal_imgs = list(set(all_imgs) - set(anno_imgs))

    for i in range(25):
        select1 = np.random.choice(anno_imgs, 1)[0]
        select2 = np.random.choice(normal_imgs, 3, replace=False).tolist()
        # 要合并的4张图，其中一张有标注
        select2.append(select1)
        # 有标注图片的位置，0表示左上角，1表示右上角，2左下角，3右下角
        anno_idx = np.random.randint(0, 4)
        if anno_idx != 3:
            select2[anno_idx], select2[3] = select2[3], select2[anno_idx]
        mosaic_img, h1, w1 = mosaic(["train/JPEGImages/" + each for each in select2])
        cv2.imwrite("train/JPEGImages/%s.png" % (i + 100), mosaic_img)
        json_file["images"].append(
            {
                "height": mosaic_img.shape[0],
                "width": mosaic_img.shape[1],
                "id": image_id,
                "file_name": "%s.png" % (i + 100)
            }
        )

        select_image_id = file_name_to_image_id(select1, json_file)
        bboxes = image_id_to_bbox(select_image_id, json_file)
        for box in bboxes:
            if anno_idx == 0:
                dh, dw = 0, 0
            elif anno_idx == 1:
                dh, dw = 0, w1 + 10
            elif anno_idx == 2:
                dh, dw = h1 + 10, 0
            else:
                dh, dw = h1 + 10, w1 + 10
            json_file["annotations"].append(
                {
                    "id": box_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        box[0] + dw,
                        box[1] + dh,
                        box[2],
                        box[3]
                    ],
                    "iscrowd": 0,
                    "area": box[2] * box[3]
                }
            )
            box_id += 1
        image_id += 1
    json.dump(json_file, open("mosaic_annotation.json", 'w'))
