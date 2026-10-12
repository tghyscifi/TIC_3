import torch
import torch.nn as nn
import matplotlib.pyplot as plt


thresh = 0.3
lens = 0.5
decay = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class firing_function(torch.autograd.Function):
    # inference
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        activtions = input.gt(thresh).float() # todo 3:define the act-function
        # activtions = input.clamp(min = 0.)
        return activtions
    # err-propagate
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        d_activtions = abs(input - thresh) < lens # todo 3:define the derivative of act-function
        # d_activtions = input.gt(0.).float()
        return grad_input * d_activtions.float()

act_fun = firing_function.apply


def neuron_block(input, last_mem, last_spike, w, b):
    mem = (last_mem * decay * (1. - last_spike)).to(device) + input.mm(w).to(device)
    spike = act_fun(mem).to(device)
    return mem, spike

class snn_n_model(nn.Module):
    def __init__(self, layer_size=None):
        super(snn_n_model, self).__init__()
        if layer_size is None:
            layer_size = []
        self.layer_size = layer_size
        self.layer_num = len(layer_size)
        print(self.layer_num)
        self.w = nn.ParameterList([torch.nn.Parameter((3 * torch.rand(layer_size[ly-1], layer_size[ly])).int().float()-1) for ly in range(1,self.layer_num)])
        self.b = nn.ParameterList([torch.nn.Parameter((0 * torch.rand(1, layer_size[ly])).int().float()) for ly in range(1,self.layer_num)])
        # for ly in range(1,self.layer_num):
        #     #print(layer_size[ly])
        #     self.w.append(torch.nn.Parameter(1e-3 * torch.rand(layer_size[ly-1], layer_size[ly])).to(device))
        #     self.b.append(torch.nn.Parameter(1e-3 * torch.rand(1, layer_size[ly])).to(device))
        #self.u1 = torch.nn.Parameter(1e-3 * torch.rand(256, 256))

    def forward(self, inputs):# inputs : [batchsize, feature_num, length_t]
        inputs = inputs.float()
        time_windows=inputs.size()[-1]
        batch_size, input_dim , input_length= inputs.size()
        #h_sumspike = []
        h_spike = []
        h_mem = []
        for ly in range (0,self.layer_num):
            # layer 0 is input
           # h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.layer_size[ly])
            #h_sumspike.append(torch.zeros(batch_size, self.layer_size[ly]))
            h_spike.append(torch.zeros(batch_size, self.layer_size[ly]))
            h_mem.append(torch.zeros(batch_size, self.layer_size[ly]))

        for step in range(time_windows):
            #(ly-1) current layer_
            #x = (inputs > torch.rand(inputs.size())).float()
            h_spike[0] = inputs[:,:,step].to(device)
            for ly in range(2,self.layer_num+1):
                h_mem[ly-1], h_spike[ly-1] = neuron_block(h_spike[ly-2], h_mem[ly-1], h_spike[ly-1], self.w[ly-2],  self.b[ly-2])

            #h_sumspike[self.layer_num-1] = h_sumspike[self.layer_num-1].to(device) + h_spike[self.layer_num-1].to(device)
            #print(h_sumspike[self.layer_num-1].shape)
        return h_spike[self.layer_num-1]


C = 4

output_channels1 = 8
batch_size = 1
output_channels2=4

class SNN_Network(nn.Module):
  def __init__(self):
    super(SNN_Network, self).__init__()

    self.snn = snn_n_model(layer_size=[output_channels2,128,128,128,128,2]).to(device)


  def forward(self, x):

    output = self.snn(x)
    print(output.size())

    return output
with torch.no_grad():
    x = torch.rand((1, 4, 37))>0.5
    model = SNN_Network()
    model.cuda()
    y0= model(x)
    input={'input':x}
    output={'output':y0}
    inout_dict={}
    inout_dict.update(input)
    inout_dict.update(output)
    torch.save(model.state_dict(),
               '../../../../../../SoftwareFiles/Wechat/WeChat Files/wxid_bsum3dnfi3gq22/FileStorage/File/2021-07/算法/parameter_SNN.pkl')
    torch.save(inout_dict,
               '../../../../../../SoftwareFiles/Wechat/WeChat Files/wxid_bsum3dnfi3gq22/FileStorage/File/2021-07/算法/parameter_SNN_inout.pkl')
    for i in model.state_dict():
        print(model.state_dict()[i])
    for i in inout_dict:
        print(i,inout_dict[i])


