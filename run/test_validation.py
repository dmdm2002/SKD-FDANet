import os
import torch
import tqdm
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.builder import model_builder

from run.scheduler import CosineAnnealingWarmUpRestarts

from utils.dataset import get_loader

from kd_losses.fdd import FeatureWiseKLD, MMDLoss
from utils.functions import save_configs, cal_metrics, get_configs
from torchmetrics.classification import ConfusionMatrix


class TestValid:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        if self.cfg['do_logging']:
            os.makedirs(f"{self.cfg['log_path']}/tensorboard_valid_2", exist_ok=True)
            self.summary = SummaryWriter(f"{self.cfg['log_path']}/tensorboard_valid_2")
            save_configs(self.cfg)

        # ---------------------------------------------------------------------------
        # ------------------------------- Build Model -------------------------------
        # ---------------------------------------------------------------------------
        self.st_model = model_builder(self.cfg)
        print(f"-------------------[Loaded Model: {self.cfg['model']}]")

        # ----------------------------------------------------------------------------------------------
        # ------------------------------- Setting Train Param and Loader -------------------------------
        # ----------------------------------------------------------------------------------------------
        self.tr_loader = get_loader(train=True,
                                    image_size=self.cfg['image_size'],
                                    crop=self.cfg['crop'],
                                    jitter=self.cfg['jitter'],
                                    noise=self.cfg['noise'],
                                    equalize=self.cfg['equalize'],
                                    injection=self.cfg['injection'],
                                    batch_size=self.cfg['batch_size'],
                                    fake_path=self.cfg['te_fake_dataset_path'],
                                    live_path=self.cfg['te_live_dataset_path'],
                                    injection_data_path=self.cfg['te_injection_fake_data_path'])

        self.loss_ce = nn.CrossEntropyLoss()
        self.feature_kd_loss = FeatureWiseKLD()
        self.mmd_loss = MMDLoss()
        self.conf_mat = ConfusionMatrix(task="binary", num_classes=2)

    def test(self):
        for ep in range(self.cfg['epoch']):
            checkpoint = torch.load(f"{self.cfg['ckp_path']}/{ep}.pth", map_location=self.cfg['device'])
            self.st_model.load_state_dict(checkpoint["model_state_dict"])

            self.st_model.eval()

            cls_xo_losses = 0
            cls_xa_losses = 0
            kd_losses = 0
            kd_mmd_losses = 0
            full_losses = 0
            tr_acc = 0

            tp, tn, fp, fn = 0, 0, 0, 0

            with torch.no_grad():
                for _, (x_o, x_a, label) in enumerate(
                        tqdm.tqdm(self.tr_loader, desc=f"[{self.cfg['model']} Test-->{ep}/{self.cfg['epoch']}]")):
                    x_o = x_o.to(self.cfg['device'])
                    x_a = x_a.to(self.cfg['device'])
                    label = label.to(self.cfg['device'])

                    res_outs_x_o, res_pool_x_o, logits_x_o = self.st_model(x_o)
                    res_outs_x_a, res_pool_x_a, logits_x_a = self.st_model(x_a)

                    kd_loss = self.feature_kd_loss(res_outs_x_a, res_outs_x_o)
                    cls_xo_loss = self.loss_ce(logits_x_o, label)
                    cls_xa_loss = self.loss_ce(logits_x_a, label)
                    kd_mmd_loss = self.mmd_loss(res_pool_x_o, res_pool_x_a)

                    full_loss = (self.cfg['fkd_lambda'] * kd_loss) + (self.cfg['mmd_lambda'] * kd_mmd_loss) + (
                            self.cfg['cls_lambda'] * (cls_xo_loss + cls_xa_loss))

                    kd_losses += kd_loss.item() * self.cfg['fkd_lambda']
                    cls_xo_losses += cls_xo_loss.item() * self.cfg['cls_lambda']
                    cls_xa_losses += cls_xa_loss.item() * self.cfg['cls_lambda']
                    kd_mmd_losses += kd_mmd_loss.item() * self.cfg['mmd_lambda']
                    full_losses += full_loss.item()

                    tr_acc += (logits_x_o.argmax(1) == label).type(torch.float).sum().item()

            cls_xo_losses = cls_xo_losses / len(self.tr_loader)
            cls_xa_losses = cls_xa_losses / len(self.tr_loader)
            kd_losses = kd_losses / len(self.tr_loader)
            kd_mmd_losses = kd_mmd_losses / len(self.tr_loader)
            full_losses = full_losses / len(self.tr_loader)

            tr_acc = tr_acc / len(self.tr_loader.dataset)

            te_acc, apcer, bpcer, acer = cal_metrics(tp, tn, fp, fn)
            self.conf_mat.reset()

            if self.cfg['do_logging']:
                self.summary.add_scalar('Valid/KD loss', kd_losses, ep)
                self.summary.add_scalar('Valid/Original Classification loss', cls_xo_losses, ep)
                self.summary.add_scalar('Valid/Shift Domain Classification loss', cls_xa_losses, ep)
                self.summary.add_scalar('Valid/MMD loss', kd_mmd_losses, ep)
                self.summary.add_scalar('Valid/Full loss', full_losses, ep)
                self.summary.add_scalar('Valid/Accuracy', tr_acc, ep)


if __name__ == '__main__':
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

    if args.dataset == 'ND':
        cross_db = 'Warsaw'
    else:
        cross_db = 'ND'

    if args.fold == '1-fold':
        cfg['te_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/2-fold/A/fake"
        cfg['te_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/A"
        cfg['tr_injection_fake_data_path'] = f"E:/dataset/Ocular/{args.dataset}/RandomInjections/PGLAV-GAN-RandomPatchInjectionAug/2-fold/A/fake"
    else:
        cfg['te_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/1-fold/B/fake"
        cfg['te_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/B"
        cfg['tr_injection_fake_data_path'] = f"E:/dataset/Ocular/{args.dataset}/RandomInjections/PGLAV-GAN-RandomPatchInjectionAug/1-fold/B/fake"

    print(
        f"Start Train: [DB: {args.dataset} || fkd_lambda: {cfg['fkd_lambda']} | cls_lambda: {cfg['cls_lambda']} | mmd_lambda: {cfg['mmd_lambda']} ||]")

    cls_train = TestValid(cfg)
    cls_train.train()
