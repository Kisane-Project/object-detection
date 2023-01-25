import os, cv2, random, json
import numpy as np
from time import time
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


def get_contour_img(data):
    # get an single image
    # img0 = data[3326]    # 392, 241, 407, 410, 3326, 2260(잼)
    image = cv2.imread(data['file_name'])

    # Get the bounding boxes
    image_bbox = np.array(data['annotations'][0]['bbox'], dtype=np.int32)

    # Crop the bounding box region from each image
    cropped_image = image[image_bbox[1]:image_bbox[3], image_bbox[0]:image_bbox[2]]

    # rect: (x, y, w, h)
    rect = (2, 2, cropped_image.shape[1] - 5, cropped_image.shape[0] - 5)
    mask = np.zeros(cropped_image.shape[:2], np.uint8)

    cv2.grabCut(cropped_image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # 0: cv2.GC_BGD, 2: cv2.GC_PR_BGD
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    grabcut = cropped_image * mask2[:, :, np.newaxis]

    img0_hsv = cv2.cvtColor(grabcut, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 1, 1])
    upper = np.array([230, 230, 230])
    range_mask = cv2.inRange(img0_hsv, lower, upper)
    contours, _ = cv2.findContours(range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grabcut_copy = np.copy(grabcut)
    initial_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = []

    for contour in initial_contours:
        area = cv2.contourArea(contour)
        if area > 4800:
            approx = contour
            contours.append(approx)

    contour_mask = cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
    contour_mask = cv2.merge((contour_mask, contour_mask, contour_mask))

    result = cv2.bitwise_and(grabcut_copy, contour_mask)

    # set background transparent
    tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(result)
    rgba = [b, g, r, alpha]
    transparent_result = cv2.merge(rgba, 4)

    return transparent_result
def main():
    json_dir = '/home/bak/Datasets/kisan_sample_data/kisan_train.json'
    with open(json_dir) as f:
        data = json.load(f)

    new_json_dir = '/home/bak/Datasets/kisan_created_data'
    os.makedirs(new_json_dir, exist_ok=True)

    num_to_create= 5
    created_images = 0

    # create #(num_to_create) of tray images
    while created_images < num_to_create:
        create_num = np.random.randint(2, 5)
        current_dir = os.getcwd()
        tray_images = os.listdir(os.path.join(current_dir, 'tray_image'))
        tray = random.choice(tray_images)
        tray_view_point = tray[5]
        tray_image = cv2.imread(os.path.join(current_dir, 'tray_image', tray))

        file_name = f"/home/bak/Projects/Datasets/kisan_created_data/output_image_{created_images}.jpg"
        height, width = tray_image.shape[:2]

        record = {}
        record["file_name"] = file_name
        record["image_id"] = data[-1]['image_id'] + created_images + 1
        record["height"] = height
        record["width"] = width

        bboxes = []

        # create #(create_num) of product images in a tray image
        for i in range(create_num):
            match = False

            # get product image same as tray image view point
            while not match:
                data_num = np.random.randint(len(data))
                file_path = data[data_num]["file_name"]
                image_name = file_path.split('/')[-1]
                view_point = image_name.split('_')[4][0]
                # TODO: 모든 파일들에서 viewpoint는 'T2' 와 같은 방식으로 파일 이름에 표기해야 함.
                if file_path.split('/')[-3] == 'TP5' and view_point == tray_view_point:
                    match = True


            product_img = get_contour_img(data[data_num])
            # cv2.imwrite("product_img.png", product_img)

            # L 460 140 -> 868 492
            # R 290 180 -> 690 525
            # T 350 140 -> 835 510
            if view_point == 'L':
                y_offset = np.random.randint(140, 490)
                x_offset = np.random.randint(460, 860)
            elif view_point == 'R':
                y_offset = np.random.randint(180, 520)
                x_offset = np.random.randint(290, 690)
            elif view_point == 'T':
                y_offset = np.random.randint(140, 510)
                x_offset = np.random.randint(350, 830)

            y1, y2 = y_offset, y_offset + product_img.shape[0]
            x1, x2 = x_offset, x_offset + product_img.shape[1]

            if y2 > height:
                y_difference = (y2 - height + 1)
                y2 = y2 - y_difference
                y1 = y1 - y_difference

            if x2 > width:
                x_difference = (x2 - width + 1)
                x2 = x2 - x_difference
                x1 = x1 - x_difference

            bbox = {
                'bbox': [x1, y1, x2, y2],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': 0
            }
            bboxes.append(bbox)

            alpha_s = product_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                tray_image[y1:y2, x1:x2, c] = (alpha_s * product_img[:, :, c] +
                                               alpha_l * tray_image[y1:y2, x1:x2, c])

        created_img_name = f"augmentation_result_{created_images}.png"
        cv2.imwrite(os.path.join(new_json_dir, created_img_name), tray_image)
        record["annotations"] = bboxes
        data.append(record)

        created_images += 1

    json_name = 'kisan_train_new.json'
    with open(os.path.join(new_json_dir, json_name), 'w') as f:
        json.dump(data, f, indent='\t')

    print("Done")


if __name__ == '__main__':
    main()