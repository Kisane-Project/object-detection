import os, cv2, json, torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor, default_argument_parser, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from datasets.custom_mapper import KisanDataMapper


'''
TODO
- 입력한 갯수만큼 inference
- ann file이 없으면 검출 결과만 저장

'''

def main(args):
    # a single image directory for inference
    image_dir = '/home/bak/Projects/Datasets/kisan_sample_data/1059/TRAY1/DE/TP3/BA000/V0_0_1_126122270991_L2_FOV090_ANG20_MIL500_LI3_TRAY1_DE_LY_TP3_BA000_1059_20221125_171231_Color.png'
    image = cv2.imread(image_dir)

    ########## detectron2 inference configs ##########
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml")
    cfg.OUTPUT_DIR = f'./ouputs'

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0075999.pth")  # path to the model you just trained
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

    # TODO: 아래에서 참조해오는 validation 파일 수정
    dataset_dir = "/home/bak/Projects/Datasets/kisan_sample_data"
    data_mapper = KisanDataMapper(data_dir=dataset_dir, split='val')
    thing_classes = data_mapper.create_classes_list()
    MetadataCatalog.get("kisan").set(thing_classes=thing_classes)

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    ########## detectron2 inference configs ##########

    save_name = 'inference_result_'+image_dir.split('/')[-1]
    inference_results_dir = os.path.join(cfg.OUTPUT_DIR, 'inference_results')
    os.makedirs(inference_results_dir, exist_ok=True)
    save_path = os.path.join(inference_results_dir, save_name)

    v = Visualizer(image[:, :, ::-1],
                MetadataCatalog.get("kisan"),
                scale=1,
                # instance_mode=ColorMode.IMAGE_BW
                )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    print('Inference result saved as: ', save_name)
    print('Saved in: ', inference_results_dir)

    ### save inference results as json file ###
    inference_result = os.path.join(inference_results_dir, 'inference_results.json')
    if os.path.isfile(inference_result):
        with open(inference_result, 'w', encoding='utf-8') as f:
            json.dump([], f, indent='\t')

    bbox = outputs["instances"]._fields['pred_boxes'].tensor[0].to(dtype=torch.uint8)
    pred_classes = outputs["instances"]._fields['pred_classes'].to(dtype=torch.uint8)
    result = {
        "image_dir": image_dir,
        "bbox": torch.Tensor.tolist(bbox),
        "pred_classes": torch.Tensor.tolist(pred_classes[0])
    }

    with open(inference_result, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.append(result)

    with open(inference_result, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent='\t')

    print('Inference done!')


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    args.num_machines = 1
    args.config_file = "configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"
    args.image_size = [1280, 720]
    
    args.resume = False


    launch(
        main(),
        args=(args,),
    )