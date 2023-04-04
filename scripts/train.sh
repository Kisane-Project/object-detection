# !/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port=100 \
                                    train.py \
                                    --config configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py \
                                    --seed 0 \
                                    --work-dir result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
                                    --launcher pytorch


CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch \
                                    --nproc_per_node=2 \
                                    --master_port=913 \
                                    train.py \
                                    --config configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py \
                                    --seed 0 \
                                    --work-dir result/kisane/sparse_rcnn_r50_fpn_1x_coco \
                                    --launcher pytorch \
                                    --auto-scale-lr \
                                    --cfg-options data.samples_per_gpu=4



CUDA_VISIBLE_DEVICES=4 python train.py \
                                    --config configs/fcos/kisane_fcos_r50_fpn_gn-head_1x.py \
                                    --seed 0 \
                                    --work-dir result/kisane/multi_aug/fcos_r50_fpn_gn-head_1x