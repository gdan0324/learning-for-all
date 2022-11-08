import json
import os
import xml.dom.minidom
from lxml import etree


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def get_bboxes(xml_dict):
    img_h = int(xml_dict["size"]["height"])
    img_w = int(xml_dict["size"]["width"])
    bboxes = []
    for obj in xml_dict["object"]:
        bndbox = obj["bndbox"]
        xmin, ymin, xmax, ymax = int(bndbox['xmin']), \
                                 int(bndbox['ymin']), \
                                 int(bndbox['xmax']), \
                                 int(bndbox['ymax'])
        bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
    return img_w, img_h, bboxes


if __name__ == '__main__':
    json_file = {
        "info": "spytensor created",
        "license": [
            "license"
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "defect"
            }
        ]
    }
    image_id = 0
    box_id = 0
    for xml_file in os.listdir("train/Annotations"):
        with open("train/Annotations/" + xml_file) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        xml_dict = parse_xml_to_dict(xml)['annotation']
        img_w, img_h, bboxes = get_bboxes(xml_dict)
        json_file['images'].append(
            {
                "height": img_h,
                "width": img_w,
                "id": image_id,
                "file_name": xml_file.split('.')[0] + '.png'
            }
        )
        for box in bboxes:
            json_file["annotations"].append(
                {
                    "id": box_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        box[0],
                        box[1],
                        box[2],
                        box[3]
                    ],
                    "iscrowd": 0,
                    "area": box[2] * box[3]
                }
            )
            box_id += 1
        image_id += 1
    json.dump(json_file, open("train_annotation.json", 'w'))
