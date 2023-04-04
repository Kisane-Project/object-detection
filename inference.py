import os
import argparse
import torch
from mmdet.models import build_detector
from mmdet.utils import rfnext_init_model, build_dp, compat_cfg, replace_cfg_vals, update_data_root
from mmcv.runner import load_checkpoint
from mmcv import Config
import mmcv
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import mvdet 
from glob import glob
import numpy as np
from utils.nms import nms


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config_path', default='result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py', help='test config file path')
    parser.add_argument('--checkpoint_path', default='result/kisane/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/epoch_4.pth', help='checkpoint_path /.pth format')
    parser.add_argument('--seed', default=0, help='seed number')
    parser.add_argument('--data_path', default='/SSDe/kisane_DB/kisane_DB_v0_3/multi_data/G04-1015/TRAY2/DR/TP4/TO270', help='data path for single image or directory containing multiple images (format: .jpg or .png)')
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--save_dir', default='visualization/', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.backends.cudnn.benchmark = True

def load_dataset(imgs, test_pipeline, device_id=0):
    datas = []
    for img in imgs:
        data = test_pipeline(img)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    # scatter to specified GPU[
    data = scatter(data, ['cuda:%d' %device_id])[0]
    return data
   
def load_pipeline(cfg):
    cfg.data.test.pipeline[0].type = 'LoadImage'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    return test_pipeline


def load_model(cfg, device_id=0, checkpoint_path=''):
    # Load pre-trained models
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    
    # CUDA set
    model = build_dp(model, 'cuda', device_ids=[device_id])
    
    # Freeze
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
    

def make_inference(model, data):
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)[0]
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)


if __name__=='__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
    # Load Arguments
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Configure
    cfg = Config.fromfile(args.config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    
    # Set Seed
    set_seed(args.seed)
    
    # Load Dataset
    if os.path.isdir(args.data_path):
        imgs = glob(os.path.join(args.data_path, '*.png')) + glob(os.path.join(args.data_path, '*.jpg')) + glob(os.path.join(args.data_path, '*.jpeg'))
    else:
        imgs = [args.data_path]

    pipeline = load_pipeline(cfg)
    model = load_model(cfg, device_id=args.gpu_id, checkpoint_path=args.checkpoint_path)
        
    # Load Modules
    for img in imgs:
        dataset = load_dataset([img], pipeline, device_id=args.gpu_id)
        
        # Inference
        results = make_inference(model, dataset)
        bbox_result = results[0][results[0][:, 4] > args.threshold]
        selected_indices = nms(bbox_result[:, :4], bbox_result[:, 4], overlap_threshold=0.5)
        bbox_result = bbox_result[selected_indices]
            
        # Visualization
        np.save(os.path.join(args.save_dir, os.path.basename(img).replace('.png', '.npy')), bbox_result) # save bbox
        show_result_pyplot(model,
                        img,
                        results,
                        score_thr=args.threshold,
                        title='result',
                        wait_time=0,
                        palette=None,
                        out_file=os.path.join(args.save_dir, os.path.basename(img))) # save image