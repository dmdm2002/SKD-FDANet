from utils.functions import get_configs
from run.test_validation import TestValid
# from run.test import Test
import argparse
import torch
import gc
import sys


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description='Proposed Model Ablation!!')
    parser.add_argument('--path', type=str, required=True, help='이미지 폴더 경로를 입력하세요')
    parser.add_argument('--model', type=str, required=True, help='모델 이름')
    parser.add_argument('--fkd_lambda', type=float, required=True, help='Feature Knowledge distillation loss weight')
    parser.add_argument('--cls_lambda', type=float, required=True, help='Classification loss weight')
    parser.add_argument('--mmd_lambda', type=float, required=True, help='Classification loss weight')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset Name')
    parser.add_argument('--fold', type=str, required=True, help='Train Fold')

    args = parser.parse_args()
    # print(f'Now Fold: {args.fold}')

    cfg = get_configs('configs/proposed_config.yml')
    cfg['model'] = f'{args.model}'
    cfg['ckp_path'] = f'{args.path}/ckp'
    cfg['log_path'] = f'{args.path}/log'
    cfg['fkd_lambda'] = args.fkd_lambda
    cfg['cls_lambda'] = args.cls_lambda
    cfg['mmd_lambda'] = args.mmd_lambda

    if args.fold == '1-fold':
        cfg['te_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/2-fold/A/fake"
        cfg['te_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/A"
        cfg['te_injection_fake_data_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN-RandomPatchInjectionAug/2-fold/A/fake"
    else:
        cfg['te_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/1-fold/B/fake"
        cfg['te_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/B"
        cfg['te_injection_fake_data_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN-RandomPatchInjectionAug/1-fold/B/fake"

    print(
        f"Start Train: [DB: {args.dataset} || fkd_lambda: {cfg['fkd_lambda']} | cls_lambda: {cfg['cls_lambda']} | mmd_lambda: {cfg['mmd_lambda']} ||]")

    cls_test = TestValid(cfg)
    cls_test.test()
