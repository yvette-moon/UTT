
from unet import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft



def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]



class MultiPeriodUNetInpainter(nn.Module):
    def __init__(self, num_vars, top_k=2):
        super().__init__()
        self.k = top_k
        self.num_vars = num_vars

        self.unet = UNet(n_channels=num_vars, n_classes=num_vars, bilinear=True)

    def forward(self, x, mask):
        """
        x: [B, L, C] 真实的完整数据 (未来部分其实不会被网络看到)
        mask: [B, L, C] 掩码 (1为历史，0为未来待预测区域)
        """
        B, L, C = x.shape

        # 施加掩码，抹除未来信息
        masked_x = x * mask

        # 使用 FFT 动态发现当前 Batch 的 top_k 个主导周期
        period_list, period_weight = FFT_for_Period(masked_x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            if L % period != 0:
                length = ((L // period) + 1) * period
                padding = torch.zeros([B, (length - L), C]).to(x.device)
                out = torch.cat([masked_x, padding], dim=1)
            else:
                length = L
                out = masked_x

            # 重塑为 2D 图像: [Batch, Channels, Height(周期数), Width(周期长度)]
            out = out.reshape(B, length // period, period, C).permute(0, 3, 1, 2).contiguous()

            # ==============================================================
            # 【新增】：动态 2D 填充 (Dynamic 2D Padding)
            # 确保 Height 和 Width 是 16 (2^4) 的倍数，防止 MaxPool2d 尺寸归零
            # ==============================================================
            H, W = out.shape[2], out.shape[3]
            pad_h = (16 - (H % 16)) % 16
            pad_w = (16 - (W % 16)) % 16

            # F.pad 参数顺序: (左填充, 右填充, 上填充, 下填充)
            if pad_h > 0 or pad_w > 0:
                out = F.pad(out, (0, pad_w, 0, pad_h))

            # 使用 2D U-Net 进行全局修复
            out = self.unet(out)

            # ==============================================================
            # 【新增】：动态 2D 裁剪 (Dynamic 2D Cropping)
            # 把为了迁就 U-Net 凑整数而补的边角料切掉，恢复真实的 H 和 W
            # ==============================================================
            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :H, :W]

            # 将修复好的 2D 图像重新展平回 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)

            # 截取掉我们最初为了凑整 1D 周期而 padding 的多余部分
            res.append(out[:, :L, :])

        res = torch.stack(res, dim=-1)  # 形状: [B, L, C, k]

        # 聚合逻辑
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, L, C, 1)

        # 通过 FFT 的振幅权重融合
        final_reconstruction = torch.sum(res * period_weight, -1)

        return final_reconstruction