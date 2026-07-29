
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

            # --- 以下重塑逻辑直接白嫖 TimesNet 源码 ---
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
            # 【终极修复】：最小尺寸与 16 倍数双重保护
            # 确保 H 和 W 至少为 4（因为有 2 次下采样），并且是 16 的倍数
            # ==============================================================
            H, W = out.shape[2], out.shape[3]

                # 如果高度或宽度小于 4，先强制用 0 填充到 4
            target_h = max(H, 4)
            target_w = max(W, 4)

                # 再凑整到 16 的倍数
            pad_h = (16 - (target_h % 16)) % 16
            pad_w = (16 - (target_w % 16)) % 16

            total_pad_h = target_h - H + pad_h
            total_pad_w = target_w - W + pad_w

            if total_pad_h > 0 or total_pad_w > 0:
                    # F.pad 参数顺序: (左, 右, 上, 下)
                out = F.pad(out, (0, total_pad_w, 0, total_pad_h))

                # 使用轻量化 2D U-Net 进行全局修复
            out = self.unet(out)

                # ==============================================================
                # 【裁剪】：精准切回原本的 H 和 W
                # ==============================================================
            if total_pad_h > 0 or total_pad_w > 0:
                out = out[:, :, :H, :W]

                # 将修复好的 2D 图像重新展平回 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            # 截取掉我们为了凑整而 padding 的多余部分，恢复到长度 L
            res.append(out[:, :L, :])

        res = torch.stack(res, dim=-1)  # 形状: [B, L, C, k]

        # --- 聚合逻辑直接白嫖 TimesNet 源码 ---
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, L, C, 1)

        # 通过 FFT 的振幅权重，将 k 个不同周期视角的重建结果融合成最终的一条线
        final_reconstruction = torch.sum(res * period_weight, -1)

        return final_reconstruction