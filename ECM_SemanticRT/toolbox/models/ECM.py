import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch
import torch.nn.functional as F
from resnet import Backbone_ResNet152_in3


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


'''U-Net Decoder Part'''
class DConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_One = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_One(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class Feature_Fusion(nn.Module):
    def __init__(self, planes=64):
        super(Feature_Fusion, self).__init__()
        self.conv = nn.Conv2d(planes * 2, planes, kernel_size=1)

    def forward(self, x_rgb, x_thermal):
        x_fused = torch.cat([x_rgb,x_thermal],dim=1)
        Fused_fea = self.conv(x_fused)
        return Fused_fea


class Complement_Awareness(nn.Module):
    def __init__(self, layer=2048, planes=64):
        super(Complement_Awareness, self).__init__()
        self.conv_r = nn.Conv2d(planes, layer, kernel_size=1)
        self.conv_t = nn.Conv2d(planes, layer, kernel_size=1)

    def forward(self, C_R, C_T, x_rgb, x_thermal):
        rgb_fused = self.conv_r(C_R) + x_rgb
        thermal_fused = self.conv_t(C_T) + x_thermal
        return rgb_fused, thermal_fused




class Decoder(nn.Module):
    def __init__(self, n_classes=9):
        super(Decoder, self).__init__()

        '''ASPP'''
        # ASPP part FOR both RGB and thermal streams
        in_dim = 64
        reduction_dim = 64
        self.ASPP1 = []
        self.ASPP2 = []
        Aspp_rates = [1, 6, 12, 18]
        for rate in Aspp_rates:
            if rate == 1:
                kernel_size = 1
                padding = 0
            else:
                kernel_size = 3
                padding = rate

            self.ASPP1.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate,
                          bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU()
            ))
            self.ASPP2.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate,
                          bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU()
            ))
        self.ASPP1.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Conv2d(in_dim, reduction_dim, 1, stride=1, bias=False),
                                           nn.BatchNorm2d(reduction_dim),
                                           nn.ReLU()
                                           ))
        self.ASPP2.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                   nn.Conv2d(in_dim, reduction_dim, 1, stride=1, bias=False),
                                                   nn.BatchNorm2d(reduction_dim),
                                                   nn.ReLU()
                                                   ))
        self.ASPP1 = nn.ModuleList(self.ASPP1)
        self.ASPP2 = nn.ModuleList(self.ASPP2)

        self.conv_low = nn.Sequential(
            nn.Conv2d(reduction_dim * 5, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.conv_high = nn.Sequential(
            nn.Conv2d(reduction_dim * 5, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(),
            nn.Dropout(0.5))


        # self.up1_decoder = Up(2048 + 1024, 64, bilinear=True)
        # self.up2_decoder = Up(64 + 512, 64, bilinear=True)
        # self.up3_decoder = Up(64 + 256, 64, bilinear=True)
        # self.up4_decoder = Up(64 + 64, 64, bilinear=True)
        self.up1_decoder = Up(64 + 64, 64, bilinear=True)
        self.up2_decoder = Up(64 + 64, 64, bilinear=True)
        self.up3_decoder = Up(64 + 64, 64, bilinear=True)
        self.up4_decoder = Up(64 + 64, 64, bilinear=True)
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.outc_supp = OutConv(64, n_classes)
        self.outc_decoder = OutConv(64, n_classes)

    def forward(self, x1, x2, x3, x4, x5):

        x = self.up1_decoder(x5, x4)
        x = self.up2_decoder(x, x3)
        x = self.up3_decoder(x, x2)

        # ASPP part
        x_size = x.size()
        out = []
        for f in self.ASPP1:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        x = self.conv_low(torch.cat(out, 1))
        supp = self.outc_supp(x)

        x = self.up4_decoder(x, x1)
        x = self.conv_decoder(x)

        # ASPP part
        x_size = x.size()
        out2 = []
        for f in self.ASPP2:
            out2.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        x = self.conv_high(torch.cat(out2, 1))
        logits = self.outc_decoder(x)

        return [logits, supp]



class Complement_Modeling(nn.Module):
    def __init__(self, planes=None, num_class=None, factor=0):
        super(Complement_Modeling, self).__init__()

        inchannel = 64
        n_classes = num_class
        self.Ups = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(planes, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.outc_rgb = OutConv(inchannel, n_classes)

        self.conv_thermal = nn.Sequential(
            nn.Conv2d(planes, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.outc_thermal = OutConv(inchannel, n_classes)

        self.conv_rgb_Complement = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.diff_rgb = OutConv(inchannel, 1)
        self.conv_rgb_out = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, n_classes, kernel_size=1))
        self.conv_thermal_Complement = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.diff_thermal = OutConv(inchannel, 1)
        self.conv_thermal_out = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, n_classes, kernel_size=1))


        self.agg_CCM_RGB = nn.Sequential(
            nn.Conv2d(inchannel * 2, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.agg_CCM_Thermal = nn.Sequential(
            nn.Conv2d(inchannel * 2, inchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))

    def forward(self, x_rgb, x_thermal):
        x_r = self.conv_rgb(x_rgb)
        logits_rgb = self.outc_rgb(x_r)

        x_t = self.conv_thermal(x_thermal)
        logits_thermal = self.outc_thermal(x_t)

        fea_rgb = x_r.detach()
        fea_thermal = x_t.detach()

        ori_fea_rgb = fea_rgb
        Complement_rgb = self.conv_rgb_Complement(fea_thermal)
        umap_rgb = self.diff_rgb(Complement_rgb)
        Med_rgb_out = self.conv_rgb_out(ori_fea_rgb + Complement_rgb)

        ori_fea_thermal = fea_thermal
        Complement_thermal = self.conv_thermal_Complement(fea_rgb)
        umap_t = self.diff_thermal(Complement_thermal)
        Med_thermal_out = self.conv_thermal_out(ori_fea_thermal + Complement_thermal)

        Fused_RGB = self.agg_CCM_RGB(torch.cat([x_r, Complement_rgb], dim=1))
        Fused_Thermal = self.agg_CCM_Thermal(torch.cat([x_t, Complement_thermal], dim=1))

        return [self.Ups(logits_rgb), self.Ups(logits_thermal), self.Ups(Med_rgb_out), self.Ups(Med_thermal_out),
                self.Ups(umap_rgb), self.Ups(umap_t)],[Complement_rgb, Complement_thermal], [Fused_RGB, Fused_Thermal]



class ECM(nn.Module):
    def __init__(self, n_classes):
        super(ECM, self).__init__()
        # RGB Stream
        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet152_in3(pretrained=True)

        # Thermal Stream
        (
            self.layer1_thermal,
            self.layer2_thermal,
            self.layer3_thermal,
            self.layer4_thermal,
            self.layer5_thermal,
        ) = Backbone_ResNet152_in3(pretrained=True)

        self.CCM5 = Complement_Modeling(planes=2048, num_class=n_classes, factor=32)
        self.CCM4 = Complement_Modeling(planes=1024, num_class=n_classes, factor=16)
        self.CCM3 = Complement_Modeling(planes=512,  num_class=n_classes, factor=8)
        self.CCM2 = Complement_Modeling(planes=256,  num_class=n_classes, factor=4)
        self.CCM1 = Complement_Modeling(planes=64,   num_class=n_classes, factor=2)

        self.CCM_aware4 = Complement_Awareness(layer=1024, planes=64)
        self.CCM_aware3 = Complement_Awareness(layer=512, planes=64)
        self.CCM_aware2 = Complement_Awareness(layer=256, planes=64)
        self.CCM_aware1 = Complement_Awareness(layer=64, planes=64)

        self.Fuse5 = Feature_Fusion(planes=64)
        self.Fuse4 = Feature_Fusion(planes=64)
        self.Fuse3 = Feature_Fusion(planes=64)
        self.Fuse2 = Feature_Fusion(planes=64)
        self.Fuse1 = Feature_Fusion(planes=64)

        self.Up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.Up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.Up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.Up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder = Decoder(n_classes=n_classes)



    def forward(self, rgb, thermal):
        x = rgb
        ir = thermal[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)     # [b_s, 3, 480, 640]

        # Layer 1
        x1 = self.layer1_rgb(x)
        ir1 = self.layer1_thermal(ir)           # [b_s, 64, 240, 320]       x2
        sideouts1, [C_R1, C_T1], [f1_rgb, f1_ir] = self.CCM1(x1, ir1)
        F1 = self.Fuse1(f1_rgb, f1_ir)

        x1, ir1 = self.CCM_aware1(C_R1, C_T1, x1, ir1)

        # Layer 2
        x2 = self.layer2_rgb(x1)
        ir2 = self.layer2_thermal(ir1)  # [b_s, 256, 120, 160]      x4
        sideouts2, [C_R2, C_T2], [f2_rgb, f2_ir] = self.CCM2(x2, ir2)
        F2 = self.Fuse2(f2_rgb, f2_ir)

        x2, ir2 = self.CCM_aware2(C_R2, C_T2, x2, ir2)

        # Layer 3
        x3 = self.layer3_rgb(x2)
        ir3 = self.layer3_thermal(ir2)  # [b_s, 512, 60, 80]        x8
        sideouts3, [C_R3, C_T3], [f3_rgb, f3_ir] = self.CCM3(x3, ir3)
        F3 = self.Fuse3(f3_rgb, f3_ir)

        x3, ir3 = self.CCM_aware3(C_R3, C_T3, x3, ir3)

        # Layer 4
        x4 = self.layer4_rgb(x3)
        ir4 = self.layer4_thermal(ir3)  # [b_s, 1024, 30, 40]       x16
        sideouts4, [C_R4, C_T4], [f4_rgb, f4_ir] = self.CCM4(x4, ir4)
        F4 = self.Fuse4(f4_rgb, f4_ir)

        x4, ir4 = self.CCM_aware4(C_R4, C_T4, x4, ir4)

        # Layer 5
        x5 = self.layer5_rgb(x4)
        ir5 = self.layer5_thermal(ir4)  # [b_s, 2048, 15, 20]       x32
        sideouts5, _, [f5_rgb, f5_ir] = self.CCM5(x5, ir5)
        F5 = self.Fuse5(f5_rgb, f5_ir)

        # print(F1.size(), F2.size(), F3.size(), F4.size(), F5.size())

        [logits, supp] = self.decoder(F1, F2, F3, F4, F5)
        supp = torch.nn.functional.interpolate(supp, scale_factor=4, mode='bilinear')
        logits = torch.nn.functional.interpolate(logits, scale_factor=2, mode='bilinear')
        output = [logits, supp]

        return [sideouts1, sideouts2, sideouts3, sideouts4, sideouts5, output]


