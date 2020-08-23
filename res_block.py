from torch import  nn
from torch.nn import Conv2d, BatchNorm2d, ReLU

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList([
            Conv2d(in_channels, out_channels, 3, stride, padding=(1, 1)),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, 3, padding=(1, 1)),
            BatchNorm2d(out_channels),
            ReLU()
        ])
        self.bypass = Conv2d(in_channels, out_channels, 3, stride, padding=(1, 1))

    def forward(self, input_tensor):
        layer_output = input_tensor
        for layer in self.layers:
            layer_output = layer(layer_output)

        bypass_output = self.bypass(input_tensor)

        return layer_output + bypass_output
