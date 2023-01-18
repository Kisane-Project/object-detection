import os
import numpy as np
import cv2
import json
import pandas as pd
from detectron2.structures import BoxMode
from tqdm import tqdm


class KisanDataMapper:
    def __init__(self, data_dir, split):
        self.json_path = None
        self.split = split
        self.dataset_dict = []
        self.data_dir = os.path.abspath(data_dir)
        self.dataset_path = data_dir
        self.missing_data = {}
        self.missing_index = 0

        kisan_json = os.path.join(self.data_dir, f'kisan_{split}.json')
        kisan_json = os.path.abspath(kisan_json)
        if os.path.exists(kisan_json) \
                and split == f'{split}':
            self.json_path = kisan_json
            print(f'Loading {split} dataset from {self.json_path}')

    def data_mapper(self):
        dataset_path = os.path.abspath(self.data_dir)
        os.makedirs(dataset_path, exist_ok=True)

        if self.json_path:
            with open(self.json_path, 'rb') as f:
                self.dataset_dict = json.load(f)
            return self.dataset_dict

        entire_list_json = os.path.join(dataset_path, 'kisan_entire_dataset.json')
        if os.path.isfile(entire_list_json):
            print('Loading entire dataset from json file...')
            with open(entire_list_json, 'r') as f:
                dataset_list = json.load(f)
        else:
            print('Creating entire dataset json file...')
            dataset_list = {
                'rgb_img_dir': [],
                'gt_dir': [],
            }

            for (root, _, files) in tqdm(os.walk(self.data_dir), desc='Loading entire dataset'):
                # if not (not files):
                if files:
                    files.sort()

                    for orientation_name in ['R2', 'T2', 'L2']:
                        record = {}
                        try:
                            rgb_name, depth_name, gt_name = self.search(files, orientation_name, root)
                        except:
                            break

                        if rgb_name is None or gt_name is None:
                            print('\033[95m' + '[Warning]' + '\033[0m')
                            print('\033[31m' + f'Check dataset. There are not paired rgb/gt data in '
                                               f'{root} or other files exist in the folder.' + '\033[0m')
                            self.missing_data[self.missing_index] = root
                            self.missing_index += 1
                            break

                        rgb_dir = os.path.join(root, rgb_name)
                        # depth_dir = os.path.join(root, depth_name)
                        gt_dir = os.path.join(root, gt_name)

                        dataset_list['rgb_img_dir'].append(rgb_dir)
                        dataset_list['gt_dir'].append(gt_dir)

            if 0 in self.missing_data:
                missing_data_log = self.data_dir + '/missing_data.json'
                with open(missing_data_log, 'w', encoding='utf-8') as f:
                    json.dump(self.missing_data, f, indent=4)


            with open(entire_list_json, 'w', encoding='utf-8') as f:
                json.dump(dataset_list, f, indent=4)


        json_dir = os.path.join(self.data_dir, f'kisan_{self.split}.json')
        data_dir = os.path.abspath(json_dir)
        if os.path.isfile(data_dir):
            with open(data_dir, 'r') as f:
                self.dataset_dict = json.load(f)
            return self.dataset_dict
        else:
            no_of_trainset = int(len(dataset_list["rgb_img_dir"]) * 0.8)

            entire_idx = np.linspace(0, len(dataset_list["rgb_img_dir"]) - 1, len(dataset_list["rgb_img_dir"]), dtype=int)
            train_idx = np.random.choice(entire_idx, no_of_trainset, replace=False)
            test_idx = np.setdiff1d(entire_idx, train_idx)
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)
            print(f'\nEntire dataset size: \t{len(dataset_list["rgb_img_dir"])}')
            print(f'Train dataset size: \t{len(train_idx)}')
            print(f'Val dataset size: \t\t{len(test_idx)}')

            train_json_dir = os.path.join(self.data_dir, f'kisan_train.json')
            val_json_dir = os.path.join(self.data_dir, f'kisan_val.json')
            if not os.path.isfile(train_json_dir):
                print('Creating train json file...')
                train_dict = self.create_json(train_idx, dataset_list, self.data_dir, 'train')
            else:
                print('train json file already exists.')
            if not os.path.isfile(val_json_dir):
                print('Creating val json file...')
                val_dict = self.create_json(test_idx, dataset_list, self.data_dir, 'val')
            else:
                print('val json file already exists.')

            if self.split == 'train':
                return train_dict
            elif self.split == 'val':
                return val_dict

        with open(json_dir, 'w', encoding='utf-8') as f:
            json.dump(self.dataset_dict, f, indent=4)

    def create_json(self, index_list, dataset_lsit,
                    data_dir, split):
        image_id = 0
        dict = []
        rgb_dirs = dataset_lsit['rgb_img_dir']
        gt_dirs = dataset_lsit['gt_dir']
        for idx in tqdm(index_list, desc=f'Creating {split} dataset'):
            record = {}

            height, width = cv2.imread(rgb_dirs[idx]).shape[:2]
            record["file_name"] = os.path.abspath(rgb_dirs[idx])
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width
            image_id += 1

            objs = []

            bbox_value = pd.read_csv(gt_dirs[idx])
            left = bbox_value["Left"][0]
            top = bbox_value["Top"][0]
            right = bbox_value["Right"][0]
            bottom = bbox_value["Bottom"][0]
            boxes = list(map(float, [left, top, right, bottom]))

            # if 1000 <= int(gt_dirs[idx].split('/')[-6]) <= 1059:
            #     category_id = int(gt_dirs[idx].split('/')[-6]) - 870
            # elif int(gt_dirs[idx].split('/')[-6]) >= 1061:
            #     category_id = int(gt_dirs[idx].split('/')[-6]) - 871
            # else:
            #     category_id = int(gt_dirs[idx].split('/')[-6]) - 1
            category_id = 0   ### for a single class
            obj = {
                "bbox": boxes,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id
            }
            objs.append(obj)
            record["annotations"] = objs
            dict.append(record)

        json_dir = os.path.join(data_dir, f'kisan_{split}.json')
        data_dir = os.path.abspath(json_dir)

        with open(data_dir, 'w', encoding='utf-8') as f:
            json.dump(dict, f, indent=4)

        return dict

    def search(self, files, name, root):
        rgb_name = None
        depth_name = None
        gt_name = None

        for item in files:
            if name in item:
                if 'Color' in item:
                    rgb_name = item
                if 'Depth' in item:
                    depth_name = item
                if 'GT' in item:
                    gt_name = item

        try:
            return rgb_name, depth_name, gt_name
        except:
            print('\033[95m' + '[Warning]' + '\033[0m')
            print('\033[31m' + f'There is no data in {root} or other files exist in the folder.' + '\033[0m')
            # os.path.join(root, files)
            self.missing_data[self.missing_index] = root
            return None

    def create_classes_list(self):
        # class_list = os.listdir(self.data_dir)
        # class_list.sort()
        # return class_list[:-3]  # remove entire, train, test json file
        ''' Using only one class "0000" '''
        return ['0000']


if __name__ == "__main__":
    data_mapper = KisanDataMapper(data_dir='/home/bak/Projects/Datasets/kisan_sample_data', split='train')
    dataset_dicts = data_mapper.data_mapper()
    class_list = data_mapper.create_classes_list()
    print('done')
