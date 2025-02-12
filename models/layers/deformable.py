"""https://github.com/chaeyeongyun/WRA-Net/blob/main/models/wra_net.py"""
import torch
import torch.nn as nn
import torchvision


class DeformableConv2d(nn.Module):
    '''Deformable convolution block'''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) in (int, tuple), "type of kernel_size must be int or tuple"
        kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # conv layer to calculate offset
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size,
                                     stride=stride, padding=(kernel_size[0] - 1) // 2, bias=True)
        # conv layer to calculate modulator
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size[0] * kernel_size[1], kernel_size=kernel_size,
                                        stride=stride, padding=(kernel_size[0] - 1) // 2, bias=True)
        # conv layers for offset and modulator must be initilaized to zero.
        self.zero_init([self.offset_conv, self.modulator_conv])

        # conv layer for deformable conv. offset and modulator will be adapted to this layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    def zero_init(self, convlayer):
        if type(convlayer) == list:
            for c in convlayer:
                nn.init.constant_(c.weight, 0.)
                nn.init.constant_(c.bias, 0.)
        else:
            nn.init.constant_(convlayer.weight, 0.)
            nn.init.constant_(convlayer.bias, 0.)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))  # modulator has (0, 1) values.
        output = torchvision.ops.deform_conv2d(input=x,
                                               offset=offset,
                                               weight=self.conv.weight,
                                               bias=self.conv.bias,
                                               stride=self.stride,
                                               padding=self.padding,
                                               dilation=self.dilation,
                                               mask=modulator)
        return output

