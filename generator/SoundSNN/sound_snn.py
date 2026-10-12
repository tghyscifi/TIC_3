import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SNNConfig:
    def __init__(self):
        self.__snn_config = {
            'fc1': {
                'thresh': 1.0,
                'decay': 255,
                'beta': 0
            },
            'fc2': {
                'thresh': 1.0,
                'decay': 255,
                'beta': 0
            },
            'fc3': {
                'thresh': 1.0,
                'decay': 255,
                'beta': 0
            },
            'fc4': {
                'thresh': 1.0,
                'decay': 255,
                'beta': 0
            }
        }

    def __getitem__(self, item):
        result = self.__snn_config.get(item)
        if result is None:
            warnings.warn('\'{}\' is not an SNN parameter!'.format(item))
        return result


class LIFactFun(nn.Module):
    def __init__(self):
        super(LIFactFun, self).__init__()

    def forward(self, input, thresh=1.0):
        fire = input.gt(thresh).float()
        return fire
            

class LIFFCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        super().__init__()
        self.thresh = config['thresh']
        self.beta = config['beta']
        self.decay = config['decay'] / 255
        self.hidden_size = hidden_size
        self.kernel = nn.Linear(input_size, hidden_size)
        self.act = LIFactFun()

    def forward(self, input_data, summary):
        self.batch_size = input_data.size(0)
        self.time_windows = input_data.size(1)
        # synaptic_input = torch.zeros(self.batch_size, self.time_windows, self.hidden_size)  # FIXME
        synaptic_input = torch.zeros(
            self.batch_size, self.time_windows, self.hidden_size,
            device=input_data.device, dtype=input_data.dtype
        )
        summary['mlp'] = {}
        for time in range(self.time_windows):
            synaptic_input[:, time, :] = self.kernel(input_data[:, time, :])
            # summary['mlp'][time] = synaptic_input[:, time, :].squeeze(0).detach()
        output = self.mem_update(synaptic_input, summary)
        return output
    
    # 输入突触连接为fc层 输入维度为[B, T, N]
    def mem_update(self, x, summary):
        output = self.__3Dmem_update(x, summary)
        return output
    
    # LIF神经元膜电位迭代部分
    def __3Dmem_update(self, x, summary):
        mem = torch.zeros_like(x[:, 0, :])
        outputs = torch.zeros_like(x)
        summary['mem'] = {}
        for t in range(self.time_windows): 
            mem = mem + x[:, t, :]
            spike = self.act(mem.clone(), self.thresh)
            output = self.act(mem.clone(), self.thresh)
            outputs[:, t, :] = output
            mem = mem * (1 - spike.detach())
            mem = mem * self.decay + self.beta
            # summary['mem'][t] = mem.squeeze(0).detach()
        return outputs


class SNN(nn.Module):
    def __init__(self, config):
        super(SNN, self).__init__()
        
        self.fc1 = LIFFCNeuron(784, 128, config['fc1'])  # 16 -> 1*28*28=784
        self.fc2 = LIFFCNeuron(128, 128, config['fc2'])
        self.fc3 = LIFFCNeuron(128, 128, config['fc3'])
        self.fc4 = LIFFCNeuron(128, 10, config['fc4'])  # 1 -> 10

        self.summary = {}

    def forward(self, x):
        self.summary['fc1'] = {}
        # self.summary['fc1']['input'] = x.squeeze(0).detach()
        # self.summary['fc1']['weight'] = self.fc1.kernel.weight.data.detach()
        # self.summary['fc1']['bias'] = self.fc1.kernel.bias.data.detach()
        x = self.fc1(x, self.summary['fc1'])
        # self.summary['fc1']['output'] = x.squeeze(0).detach()

        self.summary['fc2'] = {}
        # self.summary['fc2']['input'] = x.squeeze(0).detach()
        # self.summary['fc2']['weight'] = self.fc2.kernel.weight.data.detach()
        # self.summary['fc2']['bias'] = self.fc2.kernel.bias.data.detach()
        x = self.fc2(x, self.summary['fc2'])
        # self.summary['fc2']['output'] = x.squeeze(0).detach()

        self.summary['fc3'] = {}
        # self.summary['fc3']['input'] = x.squeeze(0).detach()
        # self.summary['fc3']['weight'] = self.fc3.kernel.weight.data.detach()
        # self.summary['fc3']['bias'] = self.fc3.kernel.bias.data.detach()        
        x = self.fc3(x, self.summary['fc3'])
        # self.summary['fc3']['output'] = x.squeeze(0).detach()

        self.summary['fc4'] = {}
        # self.summary['fc4']['input'] = x.squeeze(0).detach()
        # self.summary['fc4']['weight'] = self.fc4.kernel.weight.data.detach()
        # self.summary['fc4']['bias'] = self.fc4.kernel.bias.data.detach()
        x = self.fc4(x, self.summary['fc4'])
        # self.summary['fc4']['output'] = x.squeeze(0).detach()

        return torch.mean(x, dim=1)
    

def build_dataloaders(data_root, batch_size, num_workers, val_size):
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=False,
        transform=transform,
    )

    train_size = len(full_train) - val_size
    val_size = val_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def preprocess_mnist_batch(inputs, sequence_length):
    """
    inputs: [B, 1, 28, 28]
    -> flatten: [B, 784]
    -> unsqueeze: [B, 1, 784]
    -> repeat T times: [B, T, 784]
    """
    inputs = inputs.view(inputs.size(0), -1)          # [B, 784]
    inputs = torch.unsqueeze(inputs, dim=1)           # [B, 1, 784]
    inputs = inputs.repeat(1, sequence_length, 1)     # [B, T, 784]
    return inputs


@torch.no_grad()
def evaluate(model, loader, device, sequence_length, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        inputs = preprocess_mnist_batch(inputs, sequence_length)

        outputs = model(inputs)   # [B, 10]
        loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snn_config = SNNConfig()
    model = SNN(config=snn_config).to(device)

    ckpt = torch.load("./results/thresh_1.0_best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader = build_dataloaders(
        data_root="/your/path/to/MNIST",
        batch_size=128,
        num_workers=16,
        val_size=5000,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        sequence_length=10,
        criterion=criterion,
    )

    print(f"Test loss = {test_loss:.4f}, Test acc = {test_acc:.4f}")

    # sequence_length = 39
    # x = torch.randn((1, sequence_length, 16))
    # x = x.mul(128).round().clamp(-128, 127)

    # snn_config = SNNConfig()
    # model = SNN(config=snn_config)
    # for name, module in model.named_modules():    
    #     if isinstance(module, LIFFCNeuron):
    #         module.kernel.weight.data = module.kernel.weight.data.mul(128).round().clamp(-128, 127)
    #         module.kernel.bias.data = module.kernel.bias.data.mul(128).round().clamp(-2**31, 2**31 - 1)

    # y = model(x)
    # print(model.summary['fc1']['mlp'][0])
    # print(y)