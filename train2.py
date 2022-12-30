#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os, cv2, random
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluators, verify_results)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import Visualizer

from datasets.custom_mapper import KisanDataMapper


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    # evaluator_type : "coco"
    evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)




class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    # @classmethod
    # def build_train_loader(cls, cfg, sampler=None):
    #     return build_detection_train_loader(
    #         cfg, mapper=custom_mapper, sampler=sampler
    #     )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN = ("kisan_train",)
    cfg.DATASETS.TEST = ("kisan_val",)
    cfg.INPUT.MIN_SIZE_TRAIN = args.image_size[1]
    cfg.INPUT.MAX_SIZE_TRAIN = args.image_size[0]

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)

    cfg.SOLVER.MAX_ITER = 10000

    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.TEST.EVAL_PERIOD = 1000

    cfg.OUTPUT_DIR = f'/home/bak/Projects/kisan/ouputs'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    dataset_dir = "/home/bak/Projects/kisan/datasets/dataset"
    train_data_mapper = KisanDataMapper(data_dir=dataset_dir, split='train')
    test_data_mapper = KisanDataMapper(data_dir=dataset_dir, split='test')

    for d in ["train", "test"]:
        data_mapper = KisanDataMapper(data_dir=dataset_dir, split=d)
        DatasetCatalog.register("kisan_" + d, lambda d=d: data_mapper.data_mapper())
        DatasetCatalog.get("kisan_" + d)
        MetadataCatalog.get("kisan_" + d).set(thing_classes=data_mapper.create_classes_list())

    # kisan_metadata = MetadataCatalog.get("kisan_train")
    ### check dataset
    # dataset_dicts = data_mapper.data_mapper()
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=kisan_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("test", vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)

    args.num_classes = len(data_mapper.create_classes_list())
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    args.config_file = "configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"
    args.image_size = [1280, 720]

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        # machine_rank=args.machine_rank,
        # dist_url=args.dist_url,
        args=(args,),
    )
