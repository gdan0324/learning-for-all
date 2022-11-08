import json

json_path = r"E:\save-github\deep-learning-all\image-classification\13_tool_COCO\coco2017\annotations\instances_val2017.json"
json_labels = json.load(open(json_path, "r"))
print(json_labels["info"])
