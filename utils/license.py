"""
Check licensing for specific images from COCO
"""

import json

# Image filename (uses endswith to compare - e.g. can just use 123456.jpg for or 000000123456.jpg but 1000000123456.jpg would also match)
images = [
    # "298251.jpg",
    # "485237.jpg",
    # "185473.jpg",
    # "2053333.jpg",
    # "375469.jpg",
    # "143068.jpg",
    # "449661.jpg",
    # "460682.jpg",
    
    # "96549.jpg",
    # "490413.jpg",
    # "554266.jpg",
    # "527220.jpg",
    # "504000.jpg",
    # "68409.jpg",
    # "161044.jpg",
    # "44590.jpg",

    "375469.jpg",
    "185473.jpg",
    "460682.jpg",
    "143068.jpg",
]

coco_instances_file = "/resources/datasets/cocos2017/annotations/instances_val2017.json"

with open(coco_instances_file) as json_file:
    data = json.load(json_file)

print("Licences:")
licenses = []
for license in data['licenses']:
    print(f"\t{license['id']}\t{license['name']}")
    licenses.append(license)

for coco_image in data["images"]:
    filename = str(coco_image["file_name"])
    for image in images:
        if filename.endswith(image):
            print(f"Image: {filename}\n\tLicense: {coco_image['license']}\n\tURL: {coco_image['flickr_url']}")
