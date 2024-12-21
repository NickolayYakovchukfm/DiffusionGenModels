class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class ControlCUNet(nn.Module):
    def __init__(self, cunet):
        super().__init__()
        self.cunet = cunet
        self.in_channels = cunet.in_channels
        self.out_channels = cunet.out_channels
        self.noise_channels = cunet.noise_channels
        self.base_factor = cunet.base_factor

        factor = 2
        self.zero0 = ZeroConv2d(self.in_channels, self.base_factor)
        self.inc = self.cunet.DoubleConv(self.in_channels, self.base_factor)
        self.zero_inc = ZeroConv2d(self.base_factor, self.base_factor)
        self.down1 = self.cunet.Down(self.base_factor, 2 * self.base_factor)
        self.zero_down1 = ZeroConv2d(2 * self.base_factor, 2 * self.base_factor)
        self.down2 = self.cunet.Down(2 * self.base_factor, 4 * self.base_factor)
        self.zero_down2 = ZeroConv2d(4 * self.base_factor, 4 * self.base_factor)
        self.down3 = self.cunet.Down(4 * self.base_factor, 8 * self.base_factor)
        self.zero_down3 = ZeroConv2d(8 * self.base_factor, 8 * self.base_factor)
        self.down4 = self.cunet.Down(8 * self.base_factor, 16 * self.base_factor // factor)
        self.zero_down4 = ZeroConv2d(16 * self.base_factor // factor,
                                     16 * self.base_factor // factor)

        # важно оставить такими же названия повторяющихся модулей, чтобы копирование сработало
        misc.copy_params_and_buffers(src_module=self.cunet, dst_module=self, require_all=False)
        for param in self.cunet.parameters():
            param.requires_grad = False

    def forward(self, x, noise_labels, class_labels, cond=None):
        if cond is None:
            return self.cunet(x, noise_labels, class_labels)

        emb = self.cunet.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.cunet.map_label is not None:
            tmp = class_labels
            emb = emb + self.cunet.map_label(tmp * np.sqrt(self.cunet.map_label.in_features))

        emb = silu(self.cunet.map_layer0(emb))
        z = silu(self.cunet.map_layer1(emb)).unsqueeze(-1).unsqueeze(-1)

        x1 = self.cunet.inc(x)
        x2 = self.cunet.down1(x1)
        x3 = self.cunet.down2(x2)
        x4 = self.cunet.down3(x3)
        x5 = self.cunet.down4(x4)

        # your code here
        c0 = x + self.zero0(cond)
        c1 = self.inc(c0)
        zero1 = self.zero_inc(c1)
        c2 = self.down1(c1)
        zero2 = self.zero_down1(c2)
        c3 = self.down2(c2)
        zero3 = self.zero_down2(c3)
        c4 = self.down3(c3)
        zero4 = self.zero_down3(c4)
        c5 = self.zero_down4(self.down4(c4))

        x = self.adain1(x5 + c5, z)
        x = self.up1(x, x4 + zero4)
        x = self.adain2(x, z)
        x = self.up2(x, x3 + zero3)
        x = self.adain3(x, z)
        x = self.up3(x, x2 + zero2)
        x = self.adain4(x, z)
        x = self.up4(x, x1 + zero1)

        out = self.outc(x)
        return out
