import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureWiseKLD(nn.Module):
    def __init__(self):
        super().__init__()
        self.kd_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def feature_wise_kd(self, layer_outs_xo, layer_outs_xa):
        interval = len(layer_outs_xo)
        losses = 0
        for i in range(interval):
            losses += self.kd_div(self.log_softmax(layer_outs_xa[i]), self.log_softmax(layer_outs_xo[i]))

        loss = losses / interval

        return loss

    def forward(self, layer_outs_xo, layer_outs_xa):
        return self.feature_wise_kd(layer_outs_xo, layer_outs_xa)


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)

        return loss


class FDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_kld_loss = FeatureWiseKLD()
        self.mmd_loss = MMDLoss()

    def forward(self, source_kld, target_kld, source_mmd, target_mmd):
        kld = self.feature_kld_loss(source_kld, target_kld)
        mmd = self.mmd_loss(source_mmd, target_mmd)

        return kld, mmd

