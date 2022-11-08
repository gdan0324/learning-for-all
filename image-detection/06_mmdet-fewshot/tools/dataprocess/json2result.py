import os
import json

BASE_PATH = r'E:\Python\OutlierDetection\MyPyod\outlier-detection\stage6'
SAVE_PATH = r'C:\Users\96212\Desktop\detection-results'

JSON_PATH = os.path.join(BASE_PATH, "result.json")

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# 读取json文件
with open(os.path.join(JSON_PATH), 'r') as load_f:
    load_dict = json.load(load_f)

# 创建txt文件
for i in range(20):
    with open(os.path.join(SAVE_PATH, str(i + 1).rjust(3, '0')) + '.txt', 'w') as file:
        pass


# box [x1,x2,w,h]
def convert(size, box):
    xmax = box[0] + box[2]
    ymax = box[1] + box[3]
    return [int(box[0]), int(box[1]), int(xmax), int(ymax)]


for item in load_dict:
    name = str(item['image_id'] + 1).rjust(3, '0')
    bbox = item['bbox']
    bbox = convert([600, 600], bbox)
    score = str(round(item['score'], 6))
    category = 'defect'
    with open(os.path.join(SAVE_PATH, name + '.txt'), 'a') as file:
        file.write(category + " ")
        file.write(score + " ")
        file.write(" ".join('%s' % b for b in bbox))
        file.write('\n')
