import os, cv2
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
from datasets.custom_mapper import KisanDataMapper

image_dir = '/home/bak/Projects/Datasets/kisan_sample_data/0001/TRAY1/BR/TP1/TO000/V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY1_BR_LY_TP1_TO000_0001_20221031_153934_Color.png'
image = cv2.imread(image_dir)

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml")
cfg.OUTPUT_DIR = f'./ouputs'

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0000199.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0000199.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold

dataset_dir = "/home/bak/Projects/Datasets/kisan_sample_data"
data_mapper = KisanDataMapper(data_dir=dataset_dir, split='val')
thing_classes = data_mapper.create_classes_list()
MetadataCatalog.get("kisan").set(thing_classes=thing_classes)


predictor = DefaultPredictor(cfg)
outputs = predictor(image)

v = Visualizer(image[:, :, ::-1],
               MetadataCatalog.get("kisan"),
               scale=1,
               instance_mode=ColorMode.IMAGE_BW
               )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite('output.png', out.get_image()[:, :, ::-1])

print('Inference done!')