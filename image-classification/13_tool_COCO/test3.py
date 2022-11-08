from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# accumulate predictions from all images
# 载入coCo2017验证集标注文件
coco_true = COCO(annotation_file=r"E:\save-github\deep-learning-all\image-classification\13_tool_COCO\coco2017\annotations\instances_val2017.json")
# 载入网络在coco2017验证集上预测的结果
coco_pre = coco_true.loadRes('./predict_results.json')
coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
