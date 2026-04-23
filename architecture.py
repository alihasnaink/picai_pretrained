import torch
import torch.nn as nn

class ConvNormNonlin(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.conv     = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=True)
        self.instnorm = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.lrelu    = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    def forward(self, x):
        return self.lrelu(self.instnorm(self.conv(x)))

class StackedConvLayers(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs, kernel_size, padding):
        super().__init__()
        layers = [ConvNormNonlin(in_ch, out_ch, kernel_size, padding)]
        for _ in range(num_convs - 1):
            layers.append(ConvNormNonlin(out_ch, out_ch, kernel_size, padding))
        self.blocks = nn.Sequential(*layers)
    def forward(self, x):
        return self.blocks(x)

class Generic_UNet(nn.Module):
    MAX_FEATURES = 320

    def __init__(
        self,
        input_channels       = 3,
        base_num_features    = 32,
        num_classes          = 2,
        num_pool             = 6,
        pool_op_kernel_sizes = [[1,2,2],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]],
        conv_kernel_sizes    = [[1,3,3],[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
    ):
        super().__init__()
        self.num_pool             = num_pool
        self.pool_op_kernel_sizes = pool_op_kernel_sizes

        features = [min(base_num_features * (2**i), self.MAX_FEATURES) for i in range(num_pool + 1)]

        # Encoder (stages 0–5)
        self.conv_blocks_context = nn.ModuleList()
        in_ch = input_channels
        for i in range(num_pool):
            ks  = conv_kernel_sizes[i]
            pad = tuple(k // 2 for k in ks)
            self.conv_blocks_context.append(StackedConvLayers(in_ch, features[i], 2, ks, pad))
            in_ch = features[i]

        # Bottleneck (stage 6) — ModuleList of 2 single-conv layers
        ks  = conv_kernel_sizes[num_pool]
        pad = tuple(k // 2 for k in ks)
        self.conv_blocks_context.append(nn.ModuleList([
            StackedConvLayers(in_ch,              features[num_pool], 1, ks, pad),
            StackedConvLayers(features[num_pool], features[num_pool], 1, ks, pad),
        ]))

        # Downsampling
        self.td = nn.ModuleList([nn.MaxPool3d(pool_op_kernel_sizes[i]) for i in range(num_pool)])

        # Decoder
        self.tu                       = nn.ModuleList()
        self.conv_blocks_localization = nn.ModuleList()
        for i in range(num_pool):
            from_ch = features[num_pool - i]
            to_ch   = features[num_pool - 1 - i]
            
            # FIX 1: bias=False — checkpoint has no tu biases
            self.tu.append(nn.ConvTranspose3d(
                from_ch, to_ch,
                kernel_size = pool_op_kernel_sizes[num_pool - 1 - i],
                stride      = pool_op_kernel_sizes[num_pool - 1 - i],
                bias        = False,
            ))
            
            # FIX 2: use conv_kernel_sizes[num_pool - i], NOT [num_pool-1-i] and NOT hardcoded (3,3,3)
            ks  = conv_kernel_sizes[num_pool - i]
            pad = tuple(k // 2 for k in ks)
            self.conv_blocks_localization.append(nn.ModuleList([
                StackedConvLayers(2 * to_ch, to_ch, 1, ks, pad),
                StackedConvLayers(to_ch,     to_ch, 1, ks, pad),
            ]))

        # Seg outputs (no bias)
        self.seg_outputs = nn.ModuleList([
            nn.Conv3d(features[num_pool - 1 - i], num_classes, kernel_size=1, bias=False)
            for i in range(num_pool)
        ])

    def forward(self, x):
        skips = []
        for i in range(self.num_pool):
            x = self.conv_blocks_context[i](x)
            skips.append(x)
            x = self.td[i](x)
        for block in self.conv_blocks_context[-1]:
            x = block(x)
        seg_outputs = []
        for i in range(self.num_pool):
            x = self.tu[i](x)
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            for block in self.conv_blocks_localization[i]:
                x = block(x)
            seg_outputs.append(self.seg_outputs[i](x))
        return seg_outputs[-1]