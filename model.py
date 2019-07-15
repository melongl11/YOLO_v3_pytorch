import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

model_urls = {
    'yolo_v3': ''
}


class BasicBlock(nn.Module):

    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(int(in_channels / 2))
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2 = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += identity

        return out


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLO_v3(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, block, num_blocks, module_def, img_size=(416, 416)):
        super(YOLO_v3, self).__init__()
        self.hyp = module_def
        self.conv = self._conv_bn_relu(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.Downsample1 = self._conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, num_blocks[0])
        self.Downsample2 = self._conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1])
        self.Downsample3 = self._conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2])
        self.Downsample4 = self._conv_bn_relu(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.layer4 = self._make_layer(block, 256, num_blocks[3])
        self.Downsample5 = self._conv_bn_relu(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.layer5 = self._make_layer(block, 512, num_blocks[4])

        self.conv5_1 = self._make_conv5(1024, 512)
        self.conv1_1 = self._conv_bn_relu(512, 1024, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1, padding=0)

        self.conv_1 = self._conv_bn_relu(512, 256, 1, 1, 0)
        self.upsample1 = Upsample()

        self.conv5_2 = self._make_conv5(768, 256)
        self.conv2_1 = self._conv_bn_relu(256, 512, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, padding=0)

        self.conv_2 = self._conv_bn_relu(256, 128, 1, 1, 0)
        self.upsample2 = Upsample()

        self.conv5_3 = self._make_conv5(384, 128)
        self.conv3_1 = self._conv_bn_relu(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, padding=0)

    def _conv_bn_relu(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def _make_layer(self, block, in_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels * 2))

        return nn.Sequential(*layers)

    def _make_conv5(self, in_channels, out_channels):
        return nn.Sequential(
            self._conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            self._conv_bn_relu(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            self._conv_bn_relu(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            self._conv_bn_relu(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            self._conv_bn_relu(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """DarkNet53"""
        x = self.conv(x)

        x = self.Downsample1(x)
        x = self.layer1(x)
        x = self.Downsample2(x)
        x = self.layer2(x)
        x = self.Downsample3(x)
        x12 = self.layer3(x)
        x = self.Downsample4(x12)
        x11 = self.layer4(x)
        x = self.Downsample5(x11)
        x = self.layer5(x)

        """YOLO_v3"""
        x = self.conv5_1(x)

        y1 = self.conv1_1(x)
        y1 = self.conv1_2(y1)

        x = self.conv_1(x)
        x = self.upsample1(x)
        x = torch.cat([x, x11], 1)

        x = self.conv5_2(x)

        y2 = self.conv2_1(x)
        y2 = self.conv2_2(y2)

        x = self.conv_2(x)
        x = self.upsample2(x)
        x = torch.cat([x, x12], 1)

        x = self.conv5_3(x)

        y3 = self.conv3_1(x)
        y3 = self.conv3_2(y3)
        y1 = y1.view(x.shape[0], 3, 85, 13, 13).permute(0, 1, 3, 4, 2).contiguous()
        y2 = y2.view(x.shape[0], 3, 85, 26, 26).permute(0, 1, 3, 4, 2).contiguous()
        y3 = y3.view(x.shape[0], 3, 85, 52, 52).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:        
            return [y1, y2, y3]
        else:
            print("Validation")
            device = 'cpu'
            ngs = [13, 26, 52]
            yv_xvs = [torch.meshgrid([torch.arange(ng), torch.arange(ng)]) for ng in ngs]
            grid_xys = [torch.stack((yv_xvs[i][0], yv_xvs[i][1]), 2).to(device).float().view((1, 1, ngs[i], ngs[i], 2)) for i in range(3)]

            anchors = [[[116,90], [156,198], [373,326]], [[30,61], [62,45], [59,119]], [[10,13], [16,30], [33,23]]]
            stride = [32, 16, 8]

            for i in range(3):
                for j in range(3):
                    anchors[i][j][0] /= float(stride[i]) 
                    anchors[i][j][1] /= float(stride[i]) 
                anchors[i] = torch.from_numpy(np.asarray(anchors[i]))
                anchors[i] = anchors[i].to(device)
            anchor_whs = [anchors[i].view(1, 3, 1, 1, 2).to(device) for i in range(3)]

            outputs = [y1, y2, y3]
            result = []
            for i in range(3):
                io = outputs[i].clone()
                io = io.to(device)
                io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + grid_xys[i]
                io[..., 2:4] = torch.exp(io[..., 2:4]).double() * anchor_whs[i].double()
                io[..., 4:] = torch.sigmoid(io[..., 4:])
                io[..., :4] *= stride[i]

                io = io.view(x.shape[0], -1, 85)
                result.append([io, outputs[i]])    

            io, p = list(zip(*result))
            return torch.cat(io, 1), p

