import torch.nn as nn

class MobilenetV1Block(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 stride=1,
                 dilation=1):

        super(MobilenetV1Block, self).__init__()

        self.conv1 = nn.Conv2d(
            inp,
            inp,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=inp,
            bias=False)
        self.norm1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            inp,
            oup,
            1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False)
        self.norm2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU(inplace=True)

        self.stride = stride
        self.dilation = dilation
    
    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):

            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu2(x)

            return x
            
        out = _inner_forward(x)

        return out
        
class MobilenetV1(nn.Module):
    '''
    MobilenetV1 backbone
    '''
    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 out_channel_num=16,
                 base_channels=64,
                 channels=[[1,2],[2,4],[4,8],[8,8,8,8,8,16]],
                 output_stages=(1,2,3,5),
                 downsample_first=False):
        super(MobilenetV1, self).__init__()

        self.downsample_first = downsample_first

        self.num_stages = len(channels) + 2

        self.block_num = sum([len(i) for i in channels])
        self.stage_num = len(channels) + 2
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.out_channels = out_channel_num * base_channels
        self.output_stages = output_stages

        self.channels = [[i * base_channels for i in j] for j in channels]

        self.mnet_layers = []
        self.output_block = []
        self.output_channels = []
        self.output_strides = []

        inp = self.in_channels
        oup = self.stem_channels

        STRIDE = 1
        size = [1024, 1024]
        downsample = lambda s: [i // 2 for i in s]
        self.input_size = {}
        self.input_channel = {}

        # Add layer_1
        layer_name = 'layer_0_0'
        self.mnet_layers.append(layer_name)
        self.input_size[layer_name] = size
        self.input_channel[layer_name] = inp
        mnet_layer = nn.Sequential(
            nn.Conv2d(
                inp,
                oup,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))
        STRIDE *= 2
        self.add_module(layer_name, mnet_layer)
        if 0 in self.output_stages:
            self.output_block.append(layer_name)
            self.output_channels.append(oup)
            self.output_strides.append(STRIDE)
        size = downsample(size)

        inp = oup

        # Add backbones

        for s, stage_channels in enumerate(self.channels):
            stage = s + 1
            blocks = len(stage_channels)
            for i, channel in enumerate(stage_channels):
                oup = channel
                block = i
                if self.downsample_first:
                    stride = 2 if i == 0 else 1
                    STRIDE *= 2 if i == 0 else 1
                else:
                    stride = 2 if block == blocks - 1 else 1
                    STRIDE *= 2 if block == blocks - 1 else 1
                dilation = 1
                layer_name = f'layer_{stage}_{block}'
                self.input_size[layer_name] = size
                self.input_channel[layer_name] = inp
                size = size if stride == 1 else downsample(size)

                if block == blocks - 1 and stage in output_stages:
                    self.output_block.append(layer_name)
                    self.output_channels.append(oup)
                    self.output_strides.append(STRIDE)

                mnet_layer = MobilenetV1Block(
                    inp, 
                    oup,
                    stride,
                    dilation)
                inp = oup
                
                self.add_module(layer_name, mnet_layer)
                self.mnet_layers.append(layer_name)

        # Add last layer
        stage = self.stage_num - 1
        layer_name = f'layer_{stage}_0'
        self.input_size[layer_name] = size
        self.input_channel[layer_name] = inp
        oup = self.out_channels
        mnet_layer = MobilenetV1Block(
            inp, 
            oup,
            1,
            1)
        
        self.add_module(layer_name, mnet_layer)
        self.mnet_layers.append(layer_name)
        if stage in output_stages:
            self.output_block.append(layer_name)
            self.output_channels.append(oup)
            self.output_strides.append(STRIDE)

    def forward(self, x):
        outs = []
        for layer_name in self.mnet_layers:
            mnet_layer = getattr(self, layer_name)
            x = mnet_layer(x)
            if layer_name in self.output_block:
                outs.append(x)
        return tuple(outs)
    
    def block_list(self):
        block_list = []
        input_list = []
        input_channel = []
        for layer_name in self.mnet_layers:
            block_list.append(getattr(self, layer_name))
            input_list.append(self.input_size[layer_name])
            input_channel.append(self.input_channel[layer_name])
        return list(zip(self.mnet_layers, block_list, input_list, input_channel))

if __name__ == '__main__':
    
    model = MobilenetV1()
    with open('./tmp', 'w') as f: 
        f.write(str(model)+'\n')
        f.write('output_layer : '+str(model.output_block)+'\n')
        f.write('all_layer : '+str(model.mnet_layers)+'\n')
        f.write('output_channels: '+str(model.output_channels)+'\n')
        f.write('output_strides: '+str(model.output_strides)+'\n')
    print(model)
    print(model.output_block)
    print(model.mnet_layers)
    print(model.output_channels)
    print(model.output_strides)

    for block in model.block_list():
        print(block)
