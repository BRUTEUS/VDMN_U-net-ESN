
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionDiffWidthHeight(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionDiffWidthHeight, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True, kdim=None, vdim=None)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x_input):
        # x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x = x_input.view(-1, self.channels, x_input.shape[-1] * x_input.shape[-2]).swapaxes(1, 2)
        x_ln = self.ln(x)
        # attention_value = F.scaled_dot_product_attention(x_ln,x_ln,x_ln)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x

        # attention_value = x_ln + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels,
                                               x_input.shape[-2], x_input.shape[-1])




class SimpleLinearMapping(nn.Module):
    def __init__(self, channels):
        super(SimpleLinearMapping, self).__init__()
        self.channels = channels

        self.ff_self = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x_input):
        # x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x = x_input.view(-1, self.channels, x_input.shape[-1] * x_input.shape[-2]).swapaxes(1, 2)

        linmap_val = self.ff_self(x)
        return linmap_val.swapaxes(2, 1).view(-1, self.channels,
                                                   x_input.shape[-2], x_input.shape[-1])



## ESN (Echo State Network) block - type of liquid neural network
""""
The ESN consists of three linear layers: W_in, W_res, and W_out. 
W_in represents the input weight matrix, W_res represents the reservoir weight matrix, 
and W_out represents the output weight matrix.

The forward method processes the input data sequentially, 
updating the state of the reservoir at each time step. 
Finally, the output is obtained by applying the W_out transformation to the final reservoir state.
"""
class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, img_width, img_height):
        super(ESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.W_in = nn.Linear(input_size, reservoir_size)
        self.W_res = nn.Linear(reservoir_size, reservoir_size)
        self.W_out = nn.Linear(reservoir_size, output_size)

        self.res_in_and_out = img_width
        self.res_dim1_size = img_height

        self.W_in = nn.Linear(self.res_in_and_out, reservoir_size)
        self.W_res = nn.Linear(reservoir_size, reservoir_size)
        self.W_out = nn.Linear(reservoir_size, self.res_in_and_out)

    def forward(self, input):

        reservoir = torch.zeros((input.size(0), self.res_dim1_size, self.reservoir_size)).to("cuda")
        # commenting out since it just repeats endlessly
        # print('ESN input shape')
        # print(input.shape)
        # print()

        ## this processes each of the channels sequentially, updating the reservior each time:
        for i in range(input.size(1)):
            input_t = input[:, i, :, :]
            reservoir = torch.tanh(self.W_in(input_t) + self.W_res(reservoir))

        output = self.W_out(reservoir)
        return output.view(output.shape[0], 1, output.shape[1], output.shape[2])


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)



class DownNoTimeEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x




class UpNoTimeEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, device, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.device = device


    def forward(self, x, skip_x):
        x = self.up(x)

        ## this is hardcoded due to the particular dimensionality of the REDS dataset:
        ## input = 180x320 with desired output of 720x1280
        ## 3 downsamplings (DownNoTimeEmbed) result in 180->90->45->22

        ##F or the first upsamping, we will get up(22) = 44, but we need a dim of 45 to recover the
        ## 45 dimensions after the middle bottleneck. Thus, append a ones vector only when the result of
        ## self.up().shape[2] is 44, to achieve the target 45 dims.
        if (x.shape[2] == 44):
            ones_vec = torch.ones(x.shape[0],
                                  x.shape[1],
                                  1,
                                  x.shape[3]).to(self.device)

            x = torch.cat([x, ones_vec], dim=2)
            del(ones_vec)

        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x




class UpNoTimeEmbed8X(nn.Module):
    def __init__(self, in_channels, out_channels, device, emb_dim=256):
        super().__init__()

        self.device = device

        self.up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.up2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv2 = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )


    def forward(self, x, skip_x):
        x = self.up(x)
        skip_x = self.up2(skip_x)

        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


"""
Standard UNet implementation, except that the final output block is either an ESN block,
or a linear mapping block. This allows us to observe the effects of using an 
ESN block for output.
"""
class UNetStandard(nn.Module):
    def __init__(self, c_in=3, c_out=3, hidden_channel_dims=4,
                 use_simple_linear_mapping=True,
                 use_ESN=False,
                 reservoir_size_multiplier=100,
                 time_dim=256,
                 num_classes=None, device="cuda",
                 target_img_width=1280, target_img_height=720):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.use_simple_linear_mapping = use_simple_linear_mapping
        self.use_ESN = use_ESN
        self.reservoir_size_multiplier = reservoir_size_multiplier
        self.target_img_width = target_img_width
        self.target_img_height = target_img_height

        hidden_channel_dims2x = int(hidden_channel_dims*2)
        hidden_channel_dims4x = int(hidden_channel_dims*4)
        hidden_channel_dims8x = int(hidden_channel_dims*8)
        hidden_channel_dims16x = int(hidden_channel_dims*16)


        self.inc = DoubleConv(c_in, hidden_channel_dims)
        self.down1 = DownNoTimeEmbed(hidden_channel_dims, hidden_channel_dims2x)
        self.sa1 = SelfAttentionDiffWidthHeight(hidden_channel_dims2x)
        self.down2 = DownNoTimeEmbed(hidden_channel_dims2x, hidden_channel_dims4x)
        self.sa2 = SelfAttentionDiffWidthHeight(hidden_channel_dims4x)

        self.down3 = DownNoTimeEmbed(hidden_channel_dims4x, hidden_channel_dims4x)
        self.sa3 = SelfAttentionDiffWidthHeight(hidden_channel_dims4x)


        self.bot1 = DoubleConv(hidden_channel_dims4x, hidden_channel_dims8x)
        self.bot2 = DoubleConv(hidden_channel_dims8x, hidden_channel_dims8x)
        self.bot3 = DoubleConv(hidden_channel_dims8x, hidden_channel_dims4x)


        self.up1 = UpNoTimeEmbed(hidden_channel_dims8x, hidden_channel_dims2x, device)
        self.sa4 = SelfAttentionDiffWidthHeight(hidden_channel_dims2x)

        self.up2 = UpNoTimeEmbed(hidden_channel_dims4x, hidden_channel_dims, device)
        self.sa5 = SelfAttentionDiffWidthHeight(hidden_channel_dims)

        self.up3 = UpNoTimeEmbed8X(hidden_channel_dims2x, hidden_channel_dims, device)
        self.sa6 = SelfAttentionDiffWidthHeight(hidden_channel_dims)


        self.linMap = SimpleLinearMapping(hidden_channel_dims)

        input_size = hidden_channel_dims
        reservoir_size = int(hidden_channel_dims*self.reservoir_size_multiplier)
        output_size = hidden_channel_dims
        self.ESN1 = ESN(input_size, reservoir_size, output_size, self.target_img_width, self.target_img_height)
        self.ESN2 = ESN(input_size, reservoir_size, output_size, self.target_img_width, self.target_img_height)
        self.ESN3 = ESN(input_size, reservoir_size, output_size, self.target_img_width, self.target_img_height)
        self.outc = nn.Conv2d(hidden_channel_dims, c_out, kernel_size=1)


    def get_amount_free_memory(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        f_mb = f / 1024 ** 2
        print("CURRENT FREE MEMORY AMOUNT = {:.3f}".format(f_mb))


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc



    def forward(self, x):

        # print("x shape")
        # print(x.shape)
        x1 = self.inc(x)
        # print("x1 shape")
        # print(x1.shape)
        # self.get_amount_free_memory()


        ## DOWNSAMPLE 1
        # x2 = self.down1(x1, t)
        x2 = self.down1(x1)
        # print("x2 shape")
        # print(x2.shape)
        x2 = self.sa1(x2)
        # print("x2 shape after self attention")
        # print(x2.shape)
        # self.get_amount_free_memory()


        ## DOWNSAMPLE 2
        # x3 = self.down2(x2, t)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        # print("x3 shape")
        # print(x3.shape)
        # self.get_amount_free_memory()


        ## DOWNSAMPLE 3
        # x4 = self.down3(x3, t)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)
        # print("x4 shape before bottleneck")
        # print(x4.shape)
        # self.get_amount_free_memory()


        ## BOTTLENECK PHASE
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print("x4 shape")
        # print(x4.shape)
        # self.get_amount_free_memory()


        ## UPSAMPLE 1
        # x = self.up1(x4, x3, t)
        x = self.up1(x4, x3)
        x4.detach().to('cpu')
        x3.detach().to('cpu')
        del (x4)
        del (x3)
        # torch.cuda.empty_cache()
        x = self.sa4(x)
        # self.get_amount_free_memory()


        ## UPSAMPLE 2
        # x = self.up2(x, x2, t)
        x = self.up2(x, x2)
        # print("shape of x after up2")
        # print(x.shape)
        x2.detach().to('cpu')
        del(x2)
        # torch.cuda.empty_cache()
        x = self.sa5(x)
        # print("shape of x after sa5")
        # print(x.shape)
        # self.get_amount_free_memory()


        ## UPSAMPLE 3
        # x = self.up3(x, x1, t)
        x = self.up3(x, x1)
        x1.detach().to('cpu')
        del(x1)
        # torch.cuda.empty_cache()
        # x = self.sa6(x)
        # self.get_amount_free_memory()


        ## Final layer - use liquid neural network block or linear mapping and convolution layer for output:
        # #----------------------------------------------------------
        if(self.use_ESN):
            ESN_x1 = self.ESN1(x)
            ESN_x2 = self.ESN2(x)
            ESN_x3 = self.ESN3(x)
            x = torch.cat([ESN_x1, ESN_x2, ESN_x3], dim=1)
            return x

        else:
            x = self.linMap(x)
            # print('shape of x before out convolution:')
            # print(x.shape)
            # print()
            output = self.outc(x)
            return output
        # #----------------------------------------------------------

