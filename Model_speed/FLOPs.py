import torch
import torch.nn as nn
import sys
import json2
import numpy as np

def get_flops(model, input_shape=(3, 224, 224)):
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.autograd.Variable(
        torch.rand(*input_shape).unsqueeze(0), requires_grad=True)
    out = model(input)

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    return total_flops


def parse_darts_log(log_path, key_point='ea_acc'):
    '''
    report vaild
    '''

    collect = []
    for l in open(log_path).readlines():
        l = l.strip('/n')
        if 'args = Namespace' in l:
            collect = []
        # if 'epoch ' in l and 'lr 'in l:
        #     epoch_num = int(l.split('epoch ')[-1].split(' lr')[0])
        #     if epoch_num == 0:
        #         print(l)
        #         print(epoch_num)
        #         collect = []
        if key_point in l:
            metirc = float(l.split(key_point)[-1])
            print(metirc)
            collect.append(metirc)
    print(collect)

def parse_vs(log_path):
    '''
    report vaild
    '''

    previous = []
    current = []

    for l in open(log_path).readlines():
        l = l.strip('/n')
        if 'args = Namespace' in l:
            previous = []
            current = []

        if 'previous_vs_current' in l:
            import pdb;pdb.set_trace()
            p = float(l.split('previous_vs_current')[0].split(' ')[-1])
            c = float(l.split('previous_vs_current')[-1].split(' ')[0])
            print(metirc)
            collect.append(metirc)
    print(collect)

def get_lantacy(arch=None, l_limit=8000, h_limit=15000):
    '''
    only support sfn1 oneshot
    '''
    if arch is None:
        arch = tuple(np.random.randint(4) for i in range(16))

    assert len(arch) == 16

    #lantacy_map = json2.read('/share5/ics/guyang/dataset/shufflent_oneshot_latency/shufflent_oneshot_latency.json')['map']
    lantacy_map = [[581.0, 741.0, 832.0, 1373.0], [450.0, 549.0, 781.0, 877.0], [402.0, 499.0, 515.0, 742.0], [473.0, 673.0, 647.0, 772.0], [550.0, 553.0, 739.0, 821.0], [450.0, 428.0, 551.0, 472.0], [271.0, 408.0, 405.0, 519.0], [342.0, 388.0, 472.0, 437.0], [347.0, 429.0, 483.0, 446.0], [309.0, 365.0, 481.0, 451.0], [425.0, 461.0, 495.0, 502.0], [276.0, 377.0, 434.0, 452.0], [391.0, 415.0, 413.0, 594.0], [197.0, 289.0, 274.0, 363.0], [148.0, 149.0, 301.0, 350.0], [238.0, 272.0, 221.0, 457.0]]
    stem = 4282
    classifer = 408
    limit = 12000
    
    arch_lantacy = stem + classifer
    for layer_id, ops_id in enumerate(arch):
        arch_lantacy += lantacy_map[layer_id][ops_id]

    return arch_lantacy, (arch_lantacy<h_limit and arch_lantacy>l_limit)

if __name__ == '__main__':
    log_path = sys.argv[1]
    parse_darts_log(log_path)
