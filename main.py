import time
import random
import torch
import argparse
import os
import numpy as np
import warnings

from loguru import logger

from train import train

torch.cuda.current_device()
str_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    start = time.time()
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='FND-2-SGAT-sen-fft-gate-CFND-parm-NEW', help='project name')
    parser.add_argument('--dataset', type=str, default='CFND_dataset', help='support Weibo17/Weibo21/CFND_dataset')
    parser.add_argument('--method', type=str, default='FND-2-SGAT', help='support FND-2/MCAN/BERT/ViT/FND-2-CLIP/FND-2-SGAT')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parser.add_argument('--num_epoch', type=int, default=1, help='num_epoch')
    parser.add_argument('--text_encoder', type=str, default='bert-base-chinese', help='bert-base-chinese/clip')
    parser.add_argument('--image_encoder', type=str, default='vit-base', help='vit-base/clip')
    parser.add_argument('--lr_mm', type=float, default=5e-3, help='--lr_mm')
    parser.add_argument('--lr_mm_cls', type=float, default=5e-3, help='--lr_mm_cls')
    parser.add_argument('--weight_decay_tfm', type=float, default=1e-4, help='--weight_decay_tfm')
    parser.add_argument('--weight_decay_other', type=float, default=1e-4, help='--weight_decay_other')
    parser.add_argument('--lr_patience', type=float, default=5, help='--lr_patience')
    parser.add_argument('--lr_factor', type=float, default=0.2, help='--lr_factor')
    parser.add_argument('--lr_text_tfm', type=float, default=5e-5, help='--lr_text_tfm')
    parser.add_argument('--lr_img_tfm', type=float, default=5e-4, help='--lr_img_tfm')
    parser.add_argument('--lr_img_cls', type=float, default=5e-4, help='--lr_img_cls')
    parser.add_argument('--lr_text_cls', type=float, default=5e-5, help='--lr_text_cls')
    parser.add_argument('--data_dir', type=str, default='datasets', help='data_dir')
    parser.add_argument('--train_test', type=bool, default=True, help='train+test or test only')
    parser.add_argument('--pretrained_dir', type=str, default='Pretrained', help='path to pretrained models from Hugging Face.')
    parser.add_argument('--model_save_dir', type=str, default='results/models_weigts', help='path to save model parameters.')
    parser.add_argument('--res_save_dir', type=str, default='results/plot_results', help='path to save training results.')
    parser.add_argument('--logs_dir', type=str, default='results/logs', help='path to log results.') 
    parser.add_argument('--seed', nargs='+', default=1, help='List of random seeds')

    args = parser.parse_args()
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    args.device = device
    
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    name_seed = args.name + '_' + str(args.seed)
    args.model_name = f'{args.method}-{args.dataset}-{name_seed}-{str_time}'
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{name_seed}-{str_time}.pth')
    args.best_model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{name_seed}-best.pth')
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    logger.add(os.path.join(args.logs_dir, f'{args.dataset}-{name_seed}-{str_time}.log'))
    
    logger.info(f"Pytorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDA device: {torch.cuda.current_device()}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"GPU name: {torch.cuda.get_device_name()}")
    logger.info("Current Hyper-Parameters:")
    logger.info(args)
    
    setup_seed(args.seed)
    if args.dataset in ['Weibo17','Weibo21','CFND_dataset']:
        train(args)
    else:
        logger.info('数据集无效')
    end = time.time()
    logger.info(f"Run {end - start} seconds in total！")
