
## Getting Started

### Environment Setup
Tested on Ubuntu 18.04, NVIDIA RTX 3090 with python 3.9, pytorch 1.11.0, torchvision 0.12.0, CUDA 11.7 and mmdetection 2.26.0
1. Install dependencies
       ```
       sudo apt update && sudo apt upgrade
       ```

2. Set up a python environment
    1. Create and activate conda environment
        ```bash
        conda create -n kisane python=3.9
        conda activate kisane
        ```

    2. Install [PytTorch](https://pytorch.org/get-started/locally/)
        - example (below works for above tested environment)
        ```bash
        conda install pytorch=1.11.0 torchvision=0.12.0 pytorch-cuda=11.7 -c pytorch -c nvidia
        ```

    3. Install [MMdetection](https://mmdetection.readthedocs.io/en/stable/get_started.html)
        - Install MMCV using MIM.
        ```bash
        pip install -U openmim
        mim install mmcv-full
        ```

        -  Install MMDetection
        ```bash
        git clone https://github.com/open-mmlab/mmdetection.git
        cd mmdetection
        pip install -v -e .
        ```

    4. Install other dependencies
       ```bash
       pip install opencv-python
       pip install tqdm
       ```

### Dataset Preparation
1. Prepare kisan dataset 
    ```
    <datafolder>/kisane_DB
        └── Calibration Sheet
            └── V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY1_CA_LY_TP1_TO000_Cal_Sheet_20221128_115329_Color.png
            └── V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY1_CA_LY_TP1_TO000_Cal_Sheet_20221128_115329_Depth.png
            └── V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY1_CA_LY_TP1_TO000_Cal_Sheet_20221128_115329.txt
            ...

        └── multi_data
            └── G04-0001
            └── G04-0002
            └── G04-0003
            ...

        └── single_data
            └── LI3
                    └──0001
                    └──0002
                    └──0003
                    ...   
    ```

2. Softlink the dataset folder into current directory
    ```bash
    ln -s <kisane_DB directory> ./dataset/
    # ln -s /SSDe/kisane_DB ./dataset/
    ```

3. Augmentation
    ```bash
    python mvdet/augmentation/extract_patch.py # generate single patch
    python mvdet/augmentation/multi_augmentation.py # augmentation
    ```

4. Convert dataset file into COCO type
    ```bash
    python mvdet/datasets/kisane2coco.py
    ```


### Train
- single-gpu training
    ```bash
    # Code
    CUDA_VISIBLE_DEVICES=<사용할 GPU ID> python train.py \
                                --config configs/<사용할 configure path> \
                                --seed 0 \
                                --work-dir result/<저장할 위치>

    # Example
    CUDA_VISIBLE_DEVICES=0 python train.py \
                                    --config configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py \
                                    --seed 0 \
                                    --work-dir result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco
    ```

- multi-gpu training
    ```bash
    # Code
    CUDA_VISIBLE_DEVICES=<사용할 GPU ID> python -m torch.distributed.launch \
                                        --nproc_per_node=<GPU 개수> \
                                        --master_port=<PORT 번호> \
                                        train.py \
                                        --config configs/<사용할 configure> \
                                        --seed 0 \
                                        --work-dir result/<저장할 위치> \
                                        --launcher pytorch

    # Example (4 GPU)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=100 \
                                        train.py \
                                        --config configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py \
                                        --seed 0 \
                                        --work-dir result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
                                        --launcher pytorch
    ```


### Check inference result
- After training finished,
    - configure_path = <save-directory>/~.py
    - checkpoint_path = <save-directory>/epoch_#.pth

- Make Inference
    ```bash
        python inference.py --data_path <data_path> --config_path <configure_path> --checkpoint_path <checkpoint_path> --gpu_id <gpu id> --save_dir <save directory>

        # Example 1: inference for directory (containing *.jpg, *.png)
        python inference.py --data_path dataset/samples/0001 --config_path result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py \
                            --checkpoint_path result/kisane/epoch_36.pth --gpu_id 0 --save_dir visualization/

        # Example 2: inference for single image
        python inference.py --data_path dataset/samples/0001/V0_0_1_123622270639_R2_FOV090_ANG20_MIL500_LI3_TRAY2_BR_LY_TP1_TO045_G04-0005_20230110_111409_Color.png \
                            --config_path result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py \
                            --checkpoint_path result/kisane/epoch_36.pth --gpu_id 0 --save_dir visualization/



## Authors
- **Sungho Shin**
