#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as functional


def weights_init(m):
    """
    Define how CNN and MLP initialize their initial value
    :param m: kind of networks
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class CNN(nn.Module):
    """
    CNN for image process
    input:4D tensor of a Batch which has several(B) of Picture(C*H*W),C= in_dim,B*in_dim*H*W
    output:4D tensor of B*out_dim*H*W
    """
    def __init__(self, in_dim: int = 1, out_dim: int = 256):
        super(CNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)

        x = functional.leaky_relu(x)
        x = functional.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = functional.leaky_relu(x)
        x = functional.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = functional.leaky_relu(x)
        x = functional.max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x = functional.leaky_relu(x)

        x = x.view(-1, self.out_dim)

        return x


class GRU(nn.Module):
    """
    RNN for image process,several 3D tensor in input will go through the RNN
    input:3D tensor Batch_size*TimeStep*Length
    output:3D tensor with B*TimeStep*L
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256):
        """
        Init for GRU
        :param in_dim: Dimension of input tensor(Length)
        :param hidden_dim: dimension of hidden
        """
        super(GRU, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.network = nn.GRU(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,  # The first dimension is batch_size,not time_steps
            dropout=0.2)

    def forward(self, x):
        """
        Forward Function Of CNN
        :param x: tensor of H*W*C
        :return:
        """
        x = self.network(x)
        return x


class MLP(nn.Module):
    """
    MLP of a process to predict x and y
    Input-->PreLayer-->cos-->OutputLayer-->Output
    input:2D tensor of Batch_size*in_dim
    output:2D tensor of Batch_size*output
    """
    def __init__(self, input_dim: int = 257, rate: float = 1.0, output: int = 2):
        # rate:y = cos(rate*x)
        super(MLP, self).__init__()
        self.rate = rate
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.OutputLayer = nn.Linear(256, output)
        self.apply(weights_init)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.linear3(x)
        x = torch.tanh(x)
        x = self.linear4(x)
        x = torch.cos(x*self.rate)
        x = self.OutputLayer(x)
        return x


class Traj(nn.Module):
    """
    Total Network from CostMap to Continuous Trajectory
    Input(CostMap,t,v0)-->CNN-->RNN-->MLP-->Output(x,y)
    ----------------------------------------------------------------------------
    Input:
    :param cost_map:5D torch.tensor:batch_size*number_per_stack*color*height*width
    :param t,v0: time,velocity at batch of stacks of cost_map,2D tensor:batch_size*1
    ----------------------------------------------------------------------------
    Output:
    :return y(t):Continuous Traj,2D torch.tensor:batch_size*2
    """
    def __init__(self, mid_dim=256):
        # mid_dim:middle dimension of between CNN and RNN
        super(Traj, self).__init__()
        self.cnn = CNN(in_dim=1, out_dim=mid_dim)
        # in_dim = 1 for cost_map(Gray Image)
        self.gru = GRU(hidden_dim=mid_dim)
        self.mlp = MLP(input_dim=256+2, rate=1.0, output=2)

    def forward(self, cost_map, t, v0):
        """
        Forward function from batch of cost_map to vx,vy
        :param cost_map: 5D torch.tensor:batch_size*number_per_stack*color*height*width
        :param t: time at batch of stacks of cost_map,2D tensor:batch_size*1
        :param v0: v0 at batch of stacks of cost_map,2D tensor:batch_size*1
        :return: vx&vy,2D tensor:batch_size*2
        """
        batch_size, time_steps, color, height, width = cost_map.size()
        x = cost_map.view(batch_size * time_steps, color, height, width)  # 4D tensor
        x = self.cnn(x)  # 4D tensor:batch_size_time_steps*CNN_output_Depth*CNN_output_Height*CNN_output_Width
        x = x.view(batch_size, time_steps, -1)  # 4D --> 3D,batch_size*time_steps*Length
        x, _, = self.gru(x)  # Dimension No Change,But 2nd dimension becomes "result of the x time"
        x = functional.leaky_relu(x[:, -1, :])  # 3D --> 2D:batch_size*Length

        x = torch.cat([x, t], dim=1)  # t:batch_size*1
        x = torch.cat([x, v0], dim=1)  # v0:batch_size*1
        output = self.mlp(x)
        return output


class TrajFrom2Pic(nn.Module):
    """
    Total Network from Simple Input to Continuous Trajectory
    Input(RGB image V, Local Route R,t,v0)-->Cat-->CNN-->RNN-->merge-->MLP-->Output(x,y)
    ----------------------------------------------------------------------------
    Input:
    :param V,R:5D torch.tensor:batch_size*number_per_stack*color*height*width
    :param t,v0: time,velocity at batch of stacks of cost_map,2D tensor:batch_size*1
    ----------------------------------------------------------------------------
    Output:
    :return y(t):Continuous Trajectory,2D torch.tensor:batch_size*2
    """
    def __init__(self, mid_dim: int = 256):
        # mid_dim:middle dimension of between CNN and RNN
        super(TrajFrom2Pic, self).__init__()
        self.cnn = CNN(in_dim=6, out_dim=mid_dim)
        # in_dim = 6 for RGB image and RGB route
        self.gru = GRU(hidden_dim=mid_dim)
        self.mlp = MLP(input_dim=258, rate=1.0, output=2)

    def forward(self, image, routine, t, v0):
        x = torch.cat([image, routine], dim=2)
        batch_size, time_steps, c, height, width = x.size()
        x = x.view(batch_size * time_steps, c, height, width)  # 5D --> 4D

        x = self.cnn(x)  # 4D tensor:batch_size_time_steps*CNN_output_Depth*CNN_output_Height*CNN_output_Width
        x = x.view(batch_size, time_steps, -1)  # 4D --> 3D,batch_size*time_steps*Length
        x, _, = self.gru(x)  # Dimension No Change,But 2nd dimension becomes "result of the x time"
        x = functional.leaky_relu(x[:, -1, :])  # 3D --> 2D,Dimension*Length

        x = torch.cat([x, t], dim=1)
        x = torch.cat([x, v0], dim=1)
        # Dimension remain,but Length=2*Length+2

        output = self.mlp(x)  # batch_size*Length --> batch_size*2
        return output


if __name__ == '__main__':
    a = torch.arange(2*1*3*256*256).reshape(2, 1, 3, 256, 256)
    print(a.shape)
    
    # v0 = torch.Tensor([[2], [2]])
    # print(torch.cat([a, v0], dim=1))
    # linear = MLP(input_dim=)
    # print(linear(a))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
