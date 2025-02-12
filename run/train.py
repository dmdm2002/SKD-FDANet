import os
import torch
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.builder import model_builder

from run.scheduler import CosineAnnealingWarmUpRestarts
from kd_losses import fdd, afd, at, crd, ofd, vid
from utils.dataset import get_loader
from utils.functions import save_configs, cal_metrics
from torchmetrics.classification import ConfusionMatrix


class Train:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        if self.cfg['do_logging']:
            os.makedirs(f"{self.cfg['log_path']}/tensorboard", exist_ok=True)
            self.summary = SummaryWriter(f"{self.cfg['log_path']}/tensorboard")
            save_configs(self.cfg)
        if self.cfg['do_ckp_save']:
            os.makedirs(f"{self.cfg['ckp_path']}", exist_ok=True)

        # ---------------------------------------------------------------------------
        # ------------------------------- Build Model -------------------------------
        # ---------------------------------------------------------------------------
        self.st_model = model_builder(self.cfg)
        print(f"-------------------[Loaded Model: {self.cfg['model']}]")

        # ----------------------------------------------------------------------------------------------
        # ------------------------------- Setting Train Param and Loader -------------------------------
        # ----------------------------------------------------------------------------------------------
        self.optimizer = optim.Adam(self.st_model.parameters(), self.cfg['lr'], (self.cfg['b1'], self.cfg['b2']))
        self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer, T_0=self.cfg['step_size'], T_mult=1,
                                                       eta_max=0.0001, T_up=1, gamma=self.cfg['gamma'])

        self.tr_loader = get_loader(train=True,
                                    image_size=self.cfg['image_size'],
                                    crop=self.cfg['crop'],
                                    jitter=self.cfg['jitter'],
                                    noise=self.cfg['noise'],
                                    equalize=self.cfg['equalize'],
                                    injection=self.cfg['injection'],
                                    batch_size=self.cfg['batch_size'],
                                    fake_path=self.cfg['tr_fake_dataset_path'],
                                    live_path=self.cfg['tr_live_dataset_path'],
                                    injection_data_path=self.cfg['tr_injection_fake_data_path'])

        self.te_loader = get_loader(train=False,
                                    image_size=self.cfg['image_size'],
                                    batch_size=self.cfg['batch_size'],
                                    fake_path=self.cfg['te_fake_dataset_path'],
                                    live_path=self.cfg['te_live_dataset_path'])

        self.attack_cross_loader = get_loader(train=False,
                                              image_size=self.cfg['image_size'],
                                              batch_size=self.cfg['batch_size'],
                                              fake_path=self.cfg['attack_fake_cross_dataset_path'],
                                              live_path=self.cfg['te_live_dataset_path'])

        self.attack_uvc_loader = get_loader(train=False,
                                            image_size=self.cfg['image_size'],
                                            batch_size=self.cfg['batch_size'],
                                            fake_path=self.cfg['attack_fake_UVC_dataset_path'],
                                            live_path=self.cfg['te_live_dataset_path'])

        self.attack_pp_loader = get_loader(train=False,
                                           image_size=self.cfg['image_size'],
                                           batch_size=self.cfg['batch_size'],
                                           fake_path=self.cfg['attack_fake_postprocessing_dataset_path'],
                                           live_path=self.cfg['te_live_dataset_path'])

        self.loss_ce = nn.CrossEntropyLoss()
        # self.loss_kd = fdd.FDD()
        self.loss_kd = at.AT(p=2)
        self.conf_mat = ConfusionMatrix(task="binary", num_classes=2)

    def train(self):
        best_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        attack_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        attack_uvc_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        attack_pp_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}

        for ep in range(self.cfg['epoch']):
            self.st_model.train()

            cls_xo_losses = 0
            cls_xa_losses = 0
            kd_losses = 0
            kd_mmd_losses = 0
            full_losses = 0
            tr_acc = 0

            tp, tn, fp, fn = 0, 0, 0, 0
            attack_tp, attack_tn, attack_fp, attack_fn = 0, 0, 0, 0
            att_uvc_tp, att_uvc_tn, att_uvc_fp, att_uvc_fn = 0, 0, 0, 0
            att_pp_tp, att_pp_tn, att_pp_fp, att_pp_fn = 0, 0, 0, 0

            for _, (source, target, label) in enumerate(
                    tqdm.tqdm(self.tr_loader, desc=f"[{self.cfg['model']} Train-->{ep}/{self.cfg['epoch']}]")):
                source = source.to(self.cfg['device'])
                target = target.to(self.cfg['device'])
                label = label.to(self.cfg['device'])

                dense_outs_source, pool_out_source, logits_source = self.st_model(source)
                dense_outs_target, pool_out_target, logits_target = self.st_model(target)

                # kld_loss, mmd_loss = self.loss_kd(dense_outs_source, dense_outs_target, pool_out_source, pool_out_target)
                other_kd_loss = self.loss_kd(dense_outs_source, dense_outs_target)
                cls_xo_loss = self.loss_ce(logits_source, label)
                cls_xa_loss = self.loss_ce(logits_target, label)

                # full_loss = (self.cfg['fkd_lambda'] * kld_loss) + (self.cfg['mmd_lambda'] * mmd_loss) + (
                #         self.cfg['cls_lambda'] * (cls_xo_loss + cls_xa_loss))

                full_loss = other_kd_loss + (self.cfg['cls_lambda'] * (cls_xo_loss + cls_xa_loss))

                full_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # kd_losses += kld_loss.item() * self.cfg['fkd_lambda']
                cls_xo_losses += cls_xo_loss.item() * self.cfg['cls_lambda']
                cls_xa_losses += cls_xa_loss.item() * self.cfg['cls_lambda']
                # kd_mmd_losses += mmd_loss.item() * self.cfg['mmd_lambda']
                full_losses += full_loss.item()

                tr_acc += (logits_source.argmax(1) == label).type(torch.float).sum().item()

            self.scheduler.step()

            with torch.no_grad():
                self.st_model.eval()
                for _, (x_o, label) in enumerate(tqdm.tqdm(self.te_loader, desc=f"[Test-->{ep}/{self.cfg['epoch']}]")):
                    x_o = x_o.to(self.cfg['device'])
                    label = label.to(self.cfg['device'])

                    _, _, logit = self.st_model(x_o)
                    [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                    tp += tp_batch
                    tn += tn_batch
                    fp += fp_batch
                    fn += fn_batch

            cls_xo_losses = cls_xo_losses / len(self.tr_loader)
            cls_xa_losses = cls_xa_losses / len(self.tr_loader)
            kd_losses = kd_losses / len(self.tr_loader)
            kd_mmd_losses = kd_mmd_losses / len(self.tr_loader)
            full_losses = full_losses / len(self.tr_loader)

            tr_acc = tr_acc / len(self.tr_loader.dataset)

            te_acc, apcer, bpcer, acer = cal_metrics(tp, tn, fp, fn)
            self.conf_mat.reset()

            if best_score['acc'] <= te_acc:
                best_score['acc'] = te_acc
                best_score['apcer'] = apcer
                best_score['bpcer'] = bpcer
                best_score['acer'] = acer
                best_score['epoch'] = ep

                with torch.no_grad():
                    self.st_model.eval()
                    for _, (x_o, label) in enumerate(
                            tqdm.tqdm(self.attack_uvc_loader, desc=f"[Test-->{ep}/{self.cfg['epoch']}]")):
                        x_o = x_o.to(self.cfg['device'])
                        label = label.to(self.cfg['device'])

                        _, _, logit = self.st_model(x_o)

                        [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                        att_uvc_tp += tp_batch
                        att_uvc_tn += tn_batch
                        att_uvc_fp += fp_batch
                        att_uvc_fn += fn_batch

                attack_uvc_score['acc'], attack_uvc_score['apcer'], attack_uvc_score['bpcer'], attack_uvc_score[
                    'acer'] = cal_metrics(att_uvc_tp, att_uvc_tn, att_uvc_fp, att_uvc_fn)
                self.conf_mat.reset()

                with torch.no_grad():
                    self.st_model.eval()
                    for _, (x_o, label) in enumerate(
                            tqdm.tqdm(self.attack_cross_loader, desc=f"[Test-->{ep}/{self.cfg['epoch']}]")):
                        x_o = x_o.to(self.cfg['device'])
                        label = label.to(self.cfg['device'])

                        _, _, logit = self.st_model(x_o)

                        [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                        attack_tp += tp_batch
                        attack_tn += tn_batch
                        attack_fp += fp_batch
                        attack_fn += fn_batch

                attack_score['acc'], attack_score['apcer'], attack_score['bpcer'], attack_score['acer'] = cal_metrics(
                    attack_tp, attack_tn, attack_fp, attack_fn)
                self.conf_mat.reset()

                with torch.no_grad():
                    self.st_model.eval()
                    for _, (x_o, label) in enumerate(
                            tqdm.tqdm(self.attack_pp_loader, desc=f"[Test-->{ep}/{self.cfg['epoch']}]")):
                        x_o = x_o.to(self.cfg['device'])
                        label = label.to(self.cfg['device'])

                        _, _, logit = self.st_model(x_o)

                        [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                        att_pp_tp += tp_batch
                        att_pp_tn += tn_batch
                        att_pp_fp += fp_batch
                        att_pp_fn += fn_batch

                attack_pp_score['acc'], attack_pp_score['apcer'], attack_pp_score['bpcer'], attack_pp_score[
                    'acer'] = cal_metrics(
                    att_pp_tp, att_pp_tn, att_pp_fp, att_pp_fn)
                self.conf_mat.reset()

            if self.cfg['do_logging']:
                self.summary.add_scalar('Train/KLDivergence loss', kd_losses, ep)
                self.summary.add_scalar('Train/Original Classification loss', cls_xo_losses, ep)
                self.summary.add_scalar('Train/Shift Domain Classification loss', cls_xa_losses, ep)
                self.summary.add_scalar('Train/MMD loss', kd_mmd_losses, ep)
                self.summary.add_scalar('Train/Full loss', full_losses, ep)
                self.summary.add_scalar('Train/Accuracy', tr_acc, ep)

                self.summary.add_scalar('Test/Accuracy', te_acc, ep)
                self.summary.add_scalar('Test/APCER', apcer, ep)
                self.summary.add_scalar('Test/BPCER', bpcer, ep)
                self.summary.add_scalar('Test/ACER', acer, ep)

                f = open(f"{self.cfg['log_path']}/ACC_LOSS_LOG.txt", 'a', encoding='utf-8')
                f.write(
                    f'Train [epoch : {ep}\tKD Loss: {kd_losses}\tPGLAV-GAN Classification Loss: {cls_xo_losses}\tAug Classification Loss: {cls_xa_losses}\tMMD Loss: {kd_mmd_losses}\tAccuracy: {tr_acc}]\n')
                f.write(
                    f'Test [Accuracy: {te_acc * 100}\tAPCER: {apcer * 100}\tBPCER: {bpcer * 100}\tACER: {acer * 100}]\n')
                f.write(
                    f"Attack [Cross DB] [epoch : {best_score['epoch']} Accuracy: {attack_score['acc'] * 100}\tAPCER: {attack_score['apcer'] * 100}\tBPCER: {attack_score['bpcer'] * 100}\tACER: {attack_score['acer'] * 100}]\n")
                f.write(
                    f"Attack [UVCGAN] [epoch : {best_score['epoch']} Accuracy: {attack_uvc_score['acc'] * 100}\tAPCER: {attack_uvc_score['apcer'] * 100}\tBPCER: {attack_uvc_score['bpcer'] * 100}\tACER: {attack_uvc_score['acer'] * 100}]\n")
                f.write(
                    f"Attack [JPEG] [epoch : {best_score['epoch']} Accuracy: {attack_pp_score['acc'] * 100}\tAPCER: {attack_pp_score['apcer'] * 100}\tBPCER: {attack_pp_score['bpcer'] * 100}\tACER: {attack_pp_score['acer'] * 100}]\n")
                f.write("\n")
                f.close()

            if self.cfg['do_print']:
                print('\n')
                print('-------------------------------------------------------------------')
                print(f"Epoch: {ep}/30")
                print(
                    f"Train acc: {tr_acc} | Train kd loss: {kd_losses} | Train cls PGLAV-GAN loss: {cls_xo_losses} | Train cls Aug loss: {cls_xa_losses} | Train MMD loss: {kd_mmd_losses} | Train loss: {full_losses}")
                print(f"Test acc: {te_acc * 100}")
                print(f'APCER: {apcer * 100}  |  BPCER: {bpcer * 100}  |  ACER: {acer * 100}')
                print('-------------------------------------------------------------------')
                print(f"Best acc epoch: {best_score['epoch']}")
                print(f"Best acc: {best_score['acc']}")
                print(
                    f"APCER: {best_score['apcer'] * 100}  |  BPCER: {best_score['bpcer'] * 100}  |  ACER: {best_score['acer'] * 100}")
                print(
                    f"[Cross DB] Attack APCER: {attack_score['apcer'] * 100}  |  Attack BPCER: {attack_score['bpcer'] * 100}  |  Attack ACER: {attack_score['acer'] * 100}")
                print(
                    f"[UVCGAN] Attack APCER: {attack_uvc_score['apcer'] * 100}  |  Attack BPCER: {attack_uvc_score['bpcer'] * 100}  |  Attack ACER: {attack_uvc_score['acer'] * 100}")
                print(
                    f"[JPEG] Attack APCER: {attack_pp_score['apcer'] * 100}  |  Attack BPCER: {attack_pp_score['bpcer'] * 100}  |  Attack ACER: {attack_pp_score['acer'] * 100}")
                print('-------------------------------------------------------------------')

            if self.cfg['do_ckp_save']:
                torch.save(
                    {
                        "model_state_dict": self.st_model.state_dict(),
                        "AdamW_state_dict": self.optimizer.state_dict(),
                        "epoch": ep,
                    },
                    os.path.join(f"{self.cfg['ckp_path']}/", f"{ep}.pth"),
                )

        if self.cfg['do_logging']:
            f = open(f"{self.cfg['log_path']}/best_score.txt", 'w', encoding='utf-8')
            f.write(f'[Attack SAME GAN!!]\n')
            f.write(
                f'epoch : {best_score["epoch"]}\nAcc : {best_score["acc"] * 100}\nAPCER: {best_score["apcer"] * 100}\nBPCER: {best_score["bpcer"] * 100}\nACER: {best_score["acer"] * 100}\n')
            f.write(f'[Attack Cross DB!!]\n')
            f.write(
                f"Accuracy: {attack_score['acc'] * 100}\nAPCER: {attack_score['apcer'] * 100}\nBPCER: {attack_score['bpcer'] * 100}\nACER: {attack_score['acer'] * 100}\n")
            f.write(f'[Attack UVCGAN!!]\n')
            f.write(
                f"Accuracy: {attack_uvc_score['acc'] * 100}\nAPCER: {attack_uvc_score['apcer'] * 100}\nBPCER: {attack_uvc_score['bpcer'] * 100}\nACER: {attack_uvc_score['acer'] * 100}\n")
            f.write(f'[Attack JPEG!!]\n')
            f.write(
                f"Accuracy: {attack_pp_score['acc'] * 100}\nAPCER: {attack_pp_score['apcer'] * 100}\nBPCER: {attack_pp_score['bpcer'] * 100}\nACER: {attack_pp_score['acer'] * 100}\n")
            f.close()
