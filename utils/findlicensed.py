"""
Find n images from dataset that have a specific license
"""

import json
import random


# Number of images to find
image_count = 20

# Target license ID
target_license_id = 4

# Shuffle images (i.e. returned selection will be random)
shuffle_images = True


coco_instances_file = "/resources/datasets/cocos2017/annotations/instances_val2017.json"

with open(coco_instances_file) as json_file:
    data = json.load(json_file)

print("Licences:")
licenses = []
for license in data['licenses']:
    print(f"\t{license['id']}\t{license['name']}")
    licenses.append(license)

images = data["images"]
if shuffle_images:
    random.shuffle(images)

images_found = 0
for image in images:
    filename = str(image["file_name"])
    if image["license"] == target_license_id:
        images_found += 1
        # print(f"Image: {filename}\nLicense: {image['license']}\nCOCO URL: {image['coco_url']}\nFlickr URL: {image['flickr_url']}\n")
        # print(image['flickr_url'])
        flickr_url = image['flickr_url'].split('_')
        flickr_url = flickr_url[0].split('/')
        # print(flickr_url[len(flickr_url)-1])
        flickr_id = flickr_url[len(flickr_url)-1]
        print(f"http://flickr.com/photo.gne?id={flickr_id}")

    if images_found >= image_count:
        break