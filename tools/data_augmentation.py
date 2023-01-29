import os, cv2, random, json, argparse, time
import numpy as np
# from detectron2.structures import BoxMode
from tqdm import tqdm

global yymmdd
current_time = time.localtime()
yymmdd = time.strftime("%Y%m%d", current_time)
#https://greeksharifa.github.io/references/2021/05/18/time-datetime-usage/

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
    '''
    :param data:
    :param created_data_dir:
    :param num_to_create:
    [description]
    get a single datum and return a cropped image
    with transparent background and bounding box information

    :return: cropped transparent image, bbox
    '''

    # get a single image
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
    boundRect = [None] * len(initial_contours)

    for idx, contour in enumerate(initial_contours):
        area = cv2.contourArea(contour)
        if area > 4800:
            boundRect[idx] = cv2.boundingRect(contour)
            contours.append(contour)

    boundRect = [x for x in boundRect if x is not None]
    x1 = y1 = x2 = y2 = None

    # get entire bbox
    for idx, rect in enumerate(boundRect):
        x, y, w, h = rect
        if idx == 0:
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + w)
            y2 = max(y2, y + h)

    rect_width = x2 - x1
    rect_height = y2 - y1
    bounding_box = np.array([x1, y1, rect_width, rect_height], dtype=np.int32)

    contour_mask = cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
    contour_mask = cv2.merge((contour_mask, contour_mask, contour_mask))

    result = cv2.bitwise_and(grabcut_copy, contour_mask)

    # set background transparent
    tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(result)
    rgba = [b, g, r, alpha]
    transparent_result = cv2.merge(rgba, 4)
    product_image = transparent_result[y1:y1 + rect_height, x1:x1 + rect_width]

    return product_image, bounding_box


def get_random_augmentation(ann_data, created_data_dir, num_to_create, iou_threshold):
    '''
    :param ann_data:
    :param created_data_dir:
    :param num_to_create:
    :param iou_threshold:
    [description]
    read json file and create new json file with new images

    :return: data with dictionary format
    '''
    created_num = 0

    # there are files which have wrong name
    wrong_names = {}
    wrong_name_log = 'wrong_names.json'

    # create #(num_to_create) of tray images
    print(f"Creating {num_to_create} images...")
    while created_num < num_to_create:
        create_num = np.random.randint(2, 5)
        current_dir = os.getcwd()
        tray_images = os.listdir(os.path.join(current_dir, 'tray_image'))
        tray = random.choice(tray_images)
        tray_view_point = tray[5]
        tray_image = cv2.imread(os.path.join(current_dir, 'tray_image', tray))

        file_name = os.path.join(created_data_dir, f"output_image_{created_num}.jpg")
        height, width = tray_image.shape[:2]

        record = {}
        record["file_name"] = file_name
        record["image_id"] = ann_data[-1]['image_id'] + created_num + 1
        record["height"] = height
        record["width"] = width

        bboxes = []
        satisfied_pair = []

        # create #(create_num) of product images in a tray image
        for i in tqdm(range(create_num), desc=f"Creating {created_num} images..."):
            match = False

            # get product image same as tray image view point
            while not match:
                data_num = np.random.randint(len(ann_data))
                file_path = ann_data[data_num]["file_name"]
                image_name = file_path.split('/')[-1]
                try:
                    view_point = image_name.split('_')[4][0]
                except:
                    print('===================')
                    wrong_names[created_num] = file_path
                    with open(os.path.join(created_data_dir, wrong_name_log), 'w') as f:
                        json.dump(wrong_names, f, indent='\t')
                    print(wrong_names)
                    print('===================')
                    continue
                # TODO: 모든 파일들에서 viewpoint는 'T2' 와 같은 방식으로 파일 이름에 표기해야 함.
                if file_path.split('/')[-3] == 'TP5' and view_point == tray_view_point:
                    match = True

            try:
                # TODO: check data....
                ''' line 92 (at get_contour_img)
                    rect_width = x2 - x1
                    TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType '''
                product_img, bounding_box = get_contour_img(ann_data[data_num])
            except:
                continue
            # cv2.imwrite("product_img.png", product_img)

            # all bbox has IoU under 0.2 each other
            while not len(satisfied_pair) != len(bboxes):
                ## get random offset
                # L 460 140 -> 860 490
                # R 290 180 -> 690 520
                # T 350 140 -> 830 510
                if view_point == 'L':
                    y_offset = np.random.randint(140, 490)
                    x_offset = np.random.randint(460, 860)
                elif view_point == 'R':
                    y_offset = np.random.randint(180, 520)
                    x_offset = np.random.randint(290, 690)
                elif view_point == 'T':
                    y_offset = np.random.randint(140, 510)
                    x_offset = np.random.randint(350, 830)

                y1, y2 = y_offset, y_offset + bounding_box[3]
                x1, x2 = x_offset, x_offset + bounding_box[2]
                print('\n', y1, y2, x1, x2)
                print(y_offset, y_offset + bounding_box[3], x_offset, x_offset + bounding_box[2])

                if y2 > height:
                    y_difference = (y2 - height + 1)
                    y2 = y2 - y_difference
                    y1 = y1 - y_difference

                if x2 > width:
                    x_difference = (x2 - width + 1)
                    x2 = x2 - x_difference
                    x1 = x1 - x_difference

                if not bboxes:
                    satisfied_pair.append(True)
                else:   # check IoUs with other bboxes
                    for bbox_value in bboxes:
                        iou = IoU(bbox_value['bbox'], [x1, y1, x2, y2])
                        if iou < iou_threshold:
                            satisfied_pair.append(True)
                        else:
                            print(f"=====IoU: {iou}. Intersection area is too large.====")
                            continue

            if product_img.shape[:2] == tray_image[y1:y2, x1:x2, :].shape[:2]:
                bbox = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    # 'bbox_mode': BoxMode.XYXY_ABS,
                    'bbox_mode': None,
                    'category_id': 0
                }
                bboxes.append(bbox)

                # overlay transparent product image on tray image.
                # https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy
                # https://stackoverflow.com/questions/70295194/overlay-image-on-another-image-opencv
                alpha_s = product_img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                if product_img.shape[:2] != tray_image[y1:y2, x1:x2, :].shape[:2]:
                    print('\n')
                    print(product_img.shape)
                    print(tray_image[y1:y2, x1:x2, :].shape)
                    print(f"=====Product image size is not same as tray image size.====")
                for c in range(0, 3):
                    tray_image[y1:y2, x1:x2, c] = (alpha_s * product_img[:, :, c] +
                                                   alpha_l * tray_image[y1:y2, x1:x2, c])

        created_img_name = f"augmentation_{created_num}.png"
        cv2.imwrite(os.path.join(created_data_dir, created_img_name), tray_image)
        record["annotations"] = bboxes
        ann_data.append(record)

        print(f"Created a {created_num}th image")
        created_num += 1

    return ann_data

def main(original_json, created_data_path, num_to_create, iou_threshold):
    with open(original_json) as f:
        ann_data = json.load(f)

    os.makedirs(created_data_path, exist_ok=True)

    new_data = get_random_augmentation(ann_data, created_data_path,
                                       num_to_create, iou_threshold)

    json_name = f'kisan_train_new_{yymmdd}.json'
    with open(os.path.join(created_data_path, json_name), 'w') as f:
        json.dump(new_data, f, indent='\t')

    print("Done")


if __name__ == '__main__':
    home_dir = os.path.expanduser('~')
    parser = argparse.ArgumentParser()

    # parser.add_argument('--json_dir', type=str, default=f'/SSDc/kisane_DB/V0_0_1/LI3/kisan_train.json')
    # parser.add_argument('--created_data_path', type=str, default=f'/SSDc/kisane_DB/kisan_created_data_{yymmdd}')
    parser.add_argument('--json_dir', type=str, default=f'{home_dir}/Datasets/kisan_sample_data/kisan_train.json')
    parser.add_argument('--created_data_path', type=str, default=f'{home_dir}/Datasets/kisan_created_data_{yymmdd}')
    parser.add_argument('--num_to_create', type=int, default=10)
    parser.add_argument('--iou_threshold', type=float, default=0.15)

    args = parser.parse_args()

    main(args.json_dir, args.created_data_path, args.num_to_create, args.iou_threshold)