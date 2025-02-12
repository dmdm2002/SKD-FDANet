import tqdm
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from model.dsa_resnet_encoder import ResNetEncoders
# from model.desa_resnet_encoder import ResNetEncoders
# from model.DenseNet.desa_densenet_encoder import DESADensenetEncoder
from model.DenseNet.fdesa_bandwidth_densenet_encoder import FDESADensenetEncoder
from utils.dataset import NormalDataset
from utils.functions import save_configs, get_configs
from utils.transform import transform_handler
from torchmetrics.classification import ConfusionMatrix



class Test:
    def __init__(self, cfg: dict, ep: int, ckp_fold: str):
        self.cfg = cfg

        self.st_model = FDESADensenetEncoder(self.cfg['enc_model'], self.cfg['cls_num']).to(self.cfg['device'])
        checkpoint = torch.load(f"{self.cfg['ckp_path']}/{ep}.pth", map_location=self.cfg['device'])
        print(f"{self.cfg['ckp_path']}/{ep}.pth")
        self.st_model.load_state_dict(checkpoint["model_state_dict"])

        self.conf_mat = ConfusionMatrix(task="binary", num_classes=2)
        self.ckp_fold = ckp_fold

        self.attack_power = {'Gamma': ['gamma_0.8', 'gamma_0.9', 'gamma_1.2'],
                             'Gaussian': ['blur_3', 'blur_9', 'blur_11'],
                             'JPEG': ['q_85', 'q_90', 'q_95'],
                             'Median': ['blur_3', 'blur_9', 'blur_11'],
                             'AHE': [False]}

    def test(self):
        Attacks = ['JPEG', 'AHE']

        for attack in Attacks:
            if self.ckp_fold == '1-fold':
                folder = 'A'
            else:
                folder = 'B'

            attack_power = self.attack_power[attack]
            for power in attack_power:
                attack_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
                tp, tn, fp, fn = 0, 0, 0, 0
                if power:
                    attack_path = f"E:/dataset/Ocular/Warsaw/PGLAV-GAN/Attack/{attack}/{folder}/{power}/fake"
                else:
                    attack_path = f"E:/dataset/Ocular/Warsaw/PGLAV-GAN/Attack/{attack}/{folder}/fake"

                ori_trans = transform_handler(train=False, image_size=self.cfg['image_size'])
                attack_dataset = NormalDataset(attack_path, self.cfg['te_live_dataset_path'], ori_trans)
                attack_loader = DataLoader(dataset=attack_dataset, batch_size=self.cfg['batch_size'], shuffle=False)
                with torch.no_grad():
                    self.st_model.eval()
                    for _, (x_o, label) in enumerate(
                            tqdm.tqdm(attack_loader, desc=f"[{attack}-{power}]")):
                        x_o = x_o.to(self.cfg['device'])
                        label = label.to(self.cfg['device'])

                        _, _, logit = self.st_model(x_o)
                        [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                        tp += tp_batch
                        tn += tn_batch
                        fp += fp_batch
                        fn += fn_batch

                attack_score['acc'] = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) != 0 else 0
                attack_score['apcer'] = fp / (tn + fp) if (tn + fp) != 0 else 0
                attack_score['bpcer'] = fn / (fn + tp) if (fn + tp) != 0 else 0
                attack_score['acer'] = (attack_score['apcer'] + attack_score['bpcer']) / 2

                print(f'-----------------------------[{attack}-{power}]----------------------------')
                print(f"Attack APCER: {attack_score['apcer'] * 100}  |  Attack BPCER: {attack_score['bpcer'] * 100}  |  Attack ACER: {attack_score['acer'] * 100}")
                print('-------------------------------------------------------------------')


if __name__ == '__main__':
    from utils.functions import get_configs
    # from ru import Test

    folds = ['1-fold', '2-fold']
    root = 'E:/backup/Proposed/Ablation/Warsaw/Proposed/FDESADensenetEncoder'

    # ablations = ['No_fkd', 'stem_KD', 'stem_res1_KD', 'stem_res1-2_KD', 'stem_res1-3_KD', 'stem_res1-4_KD']
    ablations = ['stem_res1-4_KD_lr_1e-4_to_1e-5']

    ckp_info_dict = {
        'stem_res1-4_KD_lr_1e-4_to_1e-5': {'1-fold': 15},
    }

    for ablation in ablations:
        for fold in folds:
            print('-------------------------------------------------------------------')
            print(f'[{ablation} | {fold}]')
            print('-------------------------------------------------------------------')
            ckp = ckp_info_dict[ablation][fold]
            fold_folder = f'{root}/{ablation}/lambda0.3_cls0.7_mmd0.3/Train_PGLAV-GAN/{fold}/'
            config_path = f'{fold_folder}/log/train_parameters.yml'

            cfg = get_configs(config_path)
            cfg['ckp_path'] = f'{root}/{ablation}/lambda0.3_cls0.7_mmd0.3/Train_PGLAV-GAN/{fold}/ckp'
            te = Test(cfg, ckp, fold)
            te.test()
            print('-------------------------------------------------------------------\n')