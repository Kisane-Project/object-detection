import cv2, random, json
import numpy as np
from detectron2.structures import BoxMode


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def main():
    json_dir = '/home/bak/Projects/Datasets/kisan_sample_data/kisan_train.json'
    with open(json_dir) as f:
        data = json.load(f)

    num_to_create = 10
    created_samples = 0
    while num_to_create > created_samples:
        # Choose a random 2 images
        imgs = np.random.choice(data, 2, replace=False)

        # Open the images
        image_1 = cv2.imread(imgs[0]['file_name'])
        image_2 = cv2.imread(imgs[1]['file_name'])

        # Get an empty tray image
        empty_tray_list = np.array(['tray_L.png', 'tray_R.png', 'tray_T.png'])
        tray_choice = np.random.choice(empty_tray_list, 1, replace=False)[0]
        tray_img = cv2.imread(f'/home/bak/Projects/kisan-electronics/tools/{tray_choice}')

        # Get the bounding boxes
        bbox_1 = np.array(imgs[0]['annotations'][0]['bbox'], dtype=np.int32)
        bbox_2 = np.array(imgs[1]['annotations'][0]['bbox'], dtype=np.int32)

        # Calculate the intersection of union
        iou = IoU(bbox_1, bbox_2)
        if iou > 0.15:
            print(f'IoU is larger than 0.15. Too many overlapping objects. Skipping this pair.')
            continue

        # Create a new image
        file_name = f"/home/bak/Projects/Datasets/kisan_created_data/output_image_{created_samples}.jpg"
        record = {}
        height, width = tray_img.shape[:2]
        record["file_name"] = file_name
        record["image_id"] = data[-1]['image_id'] + created_samples + 1
        record["height"] = height
        record["width"] = width

        # Crop the bounding box region from each image
        cropped_image_1 = image_1[bbox_1[1]:bbox_1[3], bbox_1[0]:bbox_1[2]]
        cropped_image_2 = image_2[bbox_2[1]:bbox_2[3], bbox_2[0]:bbox_2[2]]
        # cv2.imwrite('save.png', cropped_image_2)

        # Generate random x and y coordinates for the pasted images
        objs = []

        x1_1 = np.random.randint(0, image_1.shape[1] - cropped_image_1.shape[1])
        x1_2 = x1_1 + cropped_image_1.shape[1]
        y1_1 = np.random.randint(0, image_1.shape[0] - cropped_image_1.shape[0])
        y1_2 = y1_1 + cropped_image_1.shape[0]
        obj = {
            "bbox": [x1_1, y1_1, x1_2, y1_2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0
        }
        objs.append(obj)

        x2_1 = np.random.randint(0, image_1.shape[1] - cropped_image_2.shape[1])
        x2_2 = x2_1 + cropped_image_2.shape[1]
        y2_1 = np.random.randint(0, image_1.shape[0] - cropped_image_2.shape[0])
        y2_2 = y2_1 + cropped_image_2.shape[0]
        obj = {
            "bbox": [x2_1, y2_1, x2_2, y2_2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0
        }
        objs.append(obj)
        record["annotations"] = objs
        data.append(record)

        # Paste the cropped images into the new image
        tray_img[y1_1:y1_2, x1_1:x1_2] = cropped_image_1
        tray_img[y2_1:y2_2, x2_1:x2_2] = cropped_image_2

        # Save the new image
        cv2.imwrite(f"output_image_{created_samples}.jpg", tray_img)
        created_samples += 1

    with open('/home/bak/Projects/Datasets/kisan_created_data/kisan_train_new.json', 'w') as f:
        json.dump(data, f, indent='\t')



if __name__ == '__main__':
    main()