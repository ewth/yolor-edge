
"""
Evaluate results in a JSON file with COCO API
"""


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from glob import glob

annotations_file = "/resources/datasets/cocos2017/annotations/instances_val2017.json"

if __name__ == "__main__":
    annotations = COCO(annotations_file)  # init annotations api
    json_files = glob("results/*.json")

    for file in json_files:
        print(f"Evaluating {file}...")
        predictions = annotations.loadRes(file)
        eval = COCOeval(annotations, predictions, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()