import torch.nn as nn
import torch
from torch.autograd import Variable


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """
        UNetDown for image process
        input:x
        output:encoded x
        structure:Conv2d-->Norm(if chosen)-->LeakRelu-->Dropout(if chosen)
    """
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, normalize=True, leaky=0.2, dropout=0.5):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]

        if normalize:   # 是否归一化
            layers.append(nn.InstanceNorm2d(out_size))

        layers.append(nn.LeakyReLU(leaky))    # 激活函数层

        if dropout:   # 是否Dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
        UNetUp for image process
        input:x,skip_input
        output:model(x) cat skip_input
        structure:ConvTranspose2d-->Norm-->Relu-->Dropout(if chosen)
    """
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size, stride, padding, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:  # 是否Dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class LSTMCell(nn.Module):
    """
        LSTMCell for image process
        input:x
        output:model(x)
        structure:Input-->LSTM-->Output
    """
    def __init__(self, input_channels, hidden_channels, cell_num=1, layers_num=4, dropout=0.5, bias=True):
        super(LSTMCell, self).__init__()
        self.layers = nn.ModuleList([torch.nn.LSTM(input_size=input_channels, hidden_size=hidden_channels, num_layers=layers_num, bias=bias, batch_first=True, dropout=dropout)
                                     for _ in range(cell_num)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GeneratorUNet(nn.Module):
    """
        Total Network from Image、Route to Path Graph
        Input(RGB image V, Local Route R)-->UNet-->Path Graph P
        ----------------------------------------------------------------------------
        Input:
        :param V,R:4D torch.tensor:4D torch.tensor(batch_size * 2RGB * Width * Height)
        ----------------------------------------------------------------------------
        Output:
        :return P:Gray Picture of extraction from path,in order to create CostMap
    """
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorUNet, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        result = self.final(u6)

        return result,d7

class GeneratorCNN(nn.Module):
    """
        Total Network from Image、Route to Path Graph
        Input(RGB image V, Local Route R)-->UNet-->Path Graph P
        ----------------------------------------------------------------------------
        Input:
        :param V,R:4D torch.tensor:4D torch.tensor(batch_size * 2RGB * Width * Height)
        ----------------------------------------------------------------------------
        Output:
        :return P:Gray Picture of extraction from path,in order to create CostMap
    """
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorCNN, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, torch.zeros_like(d6))
        u2 = self.up2(u1, torch.zeros_like(d5))
        u3 = self.up3(u2, torch.zeros_like(d4))
        u4 = self.up4(u3, torch.zeros_like(d3))
        u5 = self.up5(u4, torch.zeros_like(d2))
        u6 = self.up6(u5, torch.zeros_like(d1))

        result = self.final(u6)

        return result,d7


class GeneratorFeedbackUNet(nn.Module):
    """
        Total Network from Image、Route to Path Graph,But with Feedback
        Input(RGB image V, Local Route R,Path Graph P{t-1})-->UNet-->Path Graph P{t}
        P{t=0} is initialized as 0
        ----------------------------------------------------------------------------
        Input:
        :param V,R,Path Graph P{t-1}:4D torch.tensor(batch_size * 2RGB+1Gray * Width * Height)
        ----------------------------------------------------------------------------
        Output:
        :return Path Graph P{t-1}:Gray Picture of extraction from path at t,in order to create CostMap
    """
    def __init__(self, in_channels=6+1, out_channels=1):
        super(GeneratorFeedbackUNet, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )
        self.h = 0
        # 定义状态变量

    def forward(self, x):
        def state_renew(input_tensor, state_tensor):
            if state_tensor == 0:  # 当torch尚未初始化
                state_tensor = torch.zeros_like(input_tensor)
            else:
                # 如果不是初始化 重整为与x相同大小的张量，不足用0填充
                state_tensor.resize_as_(torch.zeros_like(input_tensor))
            return state_tensor

        self.h = state_renew(x, self.h)  # 初始化或重整为与x相同大小的张量

        x = torch.cat((x, self.h), dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        result = self.final(u6)

        self.h = result

        return result


class GeneratorUNetLSTM(nn.Module):   # 未完成
    """
        Total Network from Image、Route to Path Graph
        Input[(RGB image V, Local Route R)*steps*batch_size]-->UNet down-->LSTM-->UNet up-->Path Graph P
        ----------------------------------------------------------------------------
        Input:
        :param (V,R)*steps*batch_size:5D torch.tensor:batch_size*steps*3D image
        ----------------------------------------------------------------------------
        Output:
        :return P:Gray Picture of extraction from path,in order to create CostMap
    """
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorUNetLSTM, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=True, dropout=0.5)

        # 定义LSTM Cell
        self.LSTM = LSTMCell(512, 512, cell_num=1, layers_num=4, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, time_steps, color, height, width = x.size()
        x = x.view(batch_size*time_steps, color, height, width)
        # DOWN
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # LSTM
        _, color, height, width = d7.size()
        x = d7.view(batch_size, time_steps, -1)
        x = LSTMCell(x)
        x = x.view(batch_size*time_steps, color, height, width)

        # UP
        u1 = self.up1(x, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up6(u6, d1)

        x = self.final(u7)
        _, color, height, width = x.size()
        result = x.view(batch_size, time_steps, color, height, width)
        return result


class Discriminator(nn.Module):
    def __init__(self, in_channels=7):
        super(Discriminator, self).__init__()

        # 定义Discriminator的单元结构
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img, img_condition):
        # Concatenate image and condition image by channels to produce input
        # print(img_B.shape)
        img_input = torch.cat((img, img_condition), 1)
        # print(img_input.shape)
        return self.model(img_input)


if __name__ == '__main__':
    print("good！")