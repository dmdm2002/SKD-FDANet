import torch.nn as nn
import torchvision.models as models

from models.layers.attentions import FrequencyDeformableAttentionBandwidth


class FDADensenetEncoder(nn.Module):
    def __init__(self, enc_model, cls_num):
        super().__init__()

        if enc_model == 'densenet121':
            encoder = models.densenet121(pretrained=True)
            self.emb_size = 1024
        elif enc_model == 'densenet161':
            encoder = models.resnet34(pretrained=True)
            self.emb_size = 1024
        elif enc_model == 'densenet169':
            encoder = models.resnet50(pretrained=True)
            self.emb_size = 1024
        else:
            print("Error encoder models!!!, You can select a models in  this list ==> [densenet121, densenet161, densenet169]")
            print("Select default Model: [densenet121]")
            encoder = models.resnet50(pretrained=True)
            self.emb_size = 1024

        self.stem = encoder.features[:4]
        self.fda_module_0 = FrequencyDeformableAttentionBandwidth(64, 4)

        self.dense_block_1 = encoder.features.denseblock1
        self.fda_module_1 = FrequencyDeformableAttentionBandwidth(256, 4)
        self.transition_1 = encoder.features.transition1

        self.dense_block_2 = encoder.features.denseblock2
        self.fda_module_2 = FrequencyDeformableAttentionBandwidth(512, 4)
        self.transition_2 = encoder.features.transition2

        self.dense_block_3 = encoder.features.denseblock3
        self.transition_3 = encoder.features.transition3

        self.dense_block_4 = encoder.features.denseblock4
        self.norm5 = encoder.features.norm5

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(in_features=self.emb_size, out_features=cls_num)

    def forward(self, x):
        x_stem = self.stem(x)
        x_attn_0 = self.fda_module_0(x_stem)

        # dense block 1
        x_dense_1 = self.dense_block_1(x_attn_0)
        x_attn_1 = self.fda_module_1(x_dense_1)
        x_trans_1 = self.transition_1(x_attn_1)  # down-sizing

        # dense block 1
        x_dense_2 = self.dense_block_2(x_trans_1)
        x_attn_2 = self.fda_module_2(x_dense_2)
        x_trans_2 = self.transition_2(x_attn_2)  # down-sizing

        # dense block 1
        x_dense_3 = self.dense_block_3(x_trans_2)
        x_trans_3 = self.transition_3(x_dense_3)  # down-sizing

        # dense block 1
        x_dense_4 = self.dense_block_4(x_trans_3)
        x_norm = self.norm5(x_dense_4)

        x_pool = self.avg_pool(x_norm)

        x_pool = x_pool.view(x_pool.size(0), -1)
        result = self.classifier(x_pool)

        return [x_stem, x_dense_1, x_dense_2, x_dense_3, x_dense_4], x_pool, result