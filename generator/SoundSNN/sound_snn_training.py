import os
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SNNConfig:
    def __init__(
        self,
        # 全局默认值
        thresh=1.0,
        decay=255,
        beta=0,
        # 各层可选覆盖
        fc1_thresh=None,
        fc1_decay=None,
        fc1_beta=None,
        fc2_thresh=None,
        fc2_decay=None,
        fc2_beta=None,
        fc3_thresh=None,
        fc3_decay=None,
        fc3_beta=None,
        fc4_thresh=None,
        fc4_decay=None,
        fc4_beta=None,
    ):
        self.__snn_config = {
            'fc1': {
                'thresh': thresh if fc1_thresh is None else fc1_thresh,
                'decay': decay if fc1_decay is None else fc1_decay,
                'beta': beta if fc1_beta is None else fc1_beta,
            },
            'fc2': {
                'thresh': thresh if fc2_thresh is None else fc2_thresh,
                'decay': decay if fc2_decay is None else fc2_decay,
                'beta': beta if fc2_beta is None else fc2_beta,
            },
            'fc3': {
                'thresh': thresh if fc3_thresh is None else fc3_thresh,
                'decay': decay if fc3_decay is None else fc3_decay,
                'beta': beta if fc3_beta is None else fc3_beta,
            },
            'fc4': {
                'thresh': thresh if fc4_thresh is None else fc4_thresh,
                'decay': decay if fc4_decay is None else fc4_decay,
                'beta': beta if fc4_beta is None else fc4_beta,
            }
        }

    def __getitem__(self, item):
        result = self.__snn_config.get(item)
        if result is None:
            warnings.warn('\'{}\' is not an SNN parameter!'.format(item))
        return result
    

class SG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = (input > 0).float()
        gamma_tensor = torch.tensor(
            [gamma], device=input.device, dtype=input.dtype
        )
        ctx.save_for_backward(input, out, gamma_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, out, gamma_tensor = ctx.saved_tensors
        gamma = gamma_tensor[0]
        grad_input = grad_output.clone()

        # triangular surrogate gradient
        tmp = (1.0 / gamma) * (1.0 / gamma) * (gamma - input.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        return grad_input, None


class LIFactFun(nn.Module):
    def __init__(self):
        super(LIFactFun, self).__init__()
        self.heaviside = SG.apply  # [NEW]

    def forward(self, input, thresh=1.0):
        # fire = input.gt(thresh).float()
        fire = self.heaviside(input - thresh, 1.0)  # [NEW]
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

        # synaptic_input = torch.zeros(self.batch_size, self.time_windows, self.hidden_size)  # [NEW]
        synaptic_input = torch.zeros(
            self.batch_size, self.time_windows, self.hidden_size,
            device=input_data.device, dtype=input_data.dtype
        )

        summary['mlp'] = {}
        for time in range(self.time_windows):
            synaptic_input[:, time, :] = self.kernel(input_data[:, time, :])
            # summary['mlp'][time] = synaptic_input[:, time, :].squeeze(0).detach()  # 存的是 [D_out, ] -> FIXME: Batch_size 不是 1 咋整？

        output = self.mem_update(synaptic_input, summary)

        return output
    
    # 输入突触连接为fc层 输入维度为[B, T, N]
    def mem_update(self, x, summary):
        output = self.__3Dmem_update(x, summary)
        return output
    
    # LIF神经元膜电位迭代部分
    def __3Dmem_update(self, x, summary):
        mem = torch.zeros_like(x[:, 0, :])  # [B, D]
        outputs = torch.zeros_like(x)  # [B, T, D]

        summary['mem'] = {}
        for t in range(self.time_windows): 
            mem = mem + x[:, t, :]

            spike = self.act(mem.clone(), self.thresh)  # FIXME: 冗余
            output = self.act(mem.clone(), self.thresh)
            outputs[:, t, :] = output

            mem = mem * (1 - spike.detach())
            mem = mem * self.decay + self.beta

            # summary['mem'][t] = mem.squeeze(0).detach()  # 存的是 [D_out, ] -> FIXME: Batch_size 不是 1 咋整？

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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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


def train_one_epoch(model, loader, optimizer, device, sequence_length, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        inputs = preprocess_mnist_batch(inputs, sequence_length)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)   # [B, 10]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data / training
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_snn_mnist')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--val-size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sequence-length', type=int, default=10)
    # fc1
    parser.add_argument('--fc1-thresh', type=float, default=None)
    parser.add_argument('--fc1-decay', type=float, default=None)
    parser.add_argument('--fc1-beta', type=float, default=None)
    # fc2
    parser.add_argument('--fc2-thresh', type=float, default=None)
    parser.add_argument('--fc2-decay', type=float, default=None)
    parser.add_argument('--fc2-beta', type=float, default=None)
    # fc3
    parser.add_argument('--fc3-thresh', type=float, default=None)
    parser.add_argument('--fc3-decay', type=float, default=None)
    parser.add_argument('--fc3-beta', type=float, default=None)
    # fc4
    parser.add_argument('--fc4-thresh', type=float, default=None)
    parser.add_argument('--fc4-decay', type=float, default=None)
    parser.add_argument('--fc4-beta', type=float, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script must run on GPU.")
    
    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
    )

    snn_config = SNNConfig(
        fc1_thresh=args.fc1_thresh,
        fc1_decay=args.fc1_decay,
        fc1_beta=args.fc1_beta,
        fc2_thresh=args.fc2_thresh,
        fc2_decay=args.fc2_decay,
        fc2_beta=args.fc2_beta,
        fc3_thresh=args.fc3_thresh,
        fc3_decay=args.fc3_decay,
        fc3_beta=args.fc3_beta,
        fc4_thresh=args.fc4_thresh,
        fc4_decay=args.fc4_decay,
        fc4_beta=args.fc4_beta,
    )
    model = SNN(config=snn_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, 'thresh_1.0_best_model.pt')
    last_path = os.path.join(args.save_dir, 'thresh_1.0_last_model.pt')

    print("Start training on GPU...")
    print(f"Device: {device}")
    print(f"Sequence length T: {args.sequence_length}")
    print(f"Save dir: {args.save_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            sequence_length=args.sequence_length,
            criterion=criterion,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            sequence_length=args.sequence_length,
            criterion=criterion,
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'args': vars(args),
                },
                best_path,
            )
            print(f"Saved best model to {best_path}")

    torch.save(
        {
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'args': vars(args),
        },
        last_path,
    )
    print(f"Saved last model to {last_path}")

    # test with best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        sequence_length=args.sequence_length,
        criterion=criterion,
    )

    print(f"Best val_acc = {best_val_acc:.4f}")
    print(f"Test loss = {test_loss:.4f}, Test acc = {test_acc:.4f}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # data / training
#     parser.add_argument('--data-root', type=str, default='./data')
#     parser.add_argument('--save-path', type=str, default=None)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--weight-decay', type=float, default=0.0)
#     parser.add_argument('--num-workers', type=int, default=16)
#     parser.add_argument('--val-size', type=int, default=5000)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--sequence-length', type=int, default=10)
#     # fc1
#     parser.add_argument('--fc1-thresh', type=float, default=None)
#     parser.add_argument('--fc1-decay', type=float, default=None)
#     parser.add_argument('--fc1-beta', type=float, default=None)
#     # fc2
#     parser.add_argument('--fc2-thresh', type=float, default=None)
#     parser.add_argument('--fc2-decay', type=float, default=None)
#     parser.add_argument('--fc2-beta', type=float, default=None)
#     # fc3
#     parser.add_argument('--fc3-thresh', type=float, default=None)
#     parser.add_argument('--fc3-decay', type=float, default=None)
#     parser.add_argument('--fc3-beta', type=float, default=None)
#     # fc4
#     parser.add_argument('--fc4-thresh', type=float, default=None)
#     parser.add_argument('--fc4-decay', type=float, default=None)
#     parser.add_argument('--fc4-beta', type=float, default=None)

#     args = parser.parse_args()

#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available. This training script must run on GPU.")
    
#     device = torch.device('cuda')
#     # os.makedirs(args.save_dir, exist_ok=True)

#     train_loader, val_loader, test_loader = build_dataloaders(
#         data_root=args.data_root,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         val_size=args.val_size,
#     )

#     snn_config = SNNConfig(
#         fc1_thresh=args.fc1_thresh,
#         fc1_decay=args.fc1_decay,
#         fc1_beta=args.fc1_beta,
#         fc2_thresh=args.fc2_thresh,
#         fc2_decay=args.fc2_decay,
#         fc2_beta=args.fc2_beta,
#         fc3_thresh=args.fc3_thresh,
#         fc3_decay=args.fc3_decay,
#         fc3_beta=args.fc3_beta,
#         fc4_thresh=args.fc4_thresh,
#         fc4_decay=args.fc4_decay,
#         fc4_beta=args.fc4_beta,
#     )
#     model = SNN(config=snn_config).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )

#     # result_txt_path = os.path.join(args.save_dir, 'epoch_acc.txt')

#     # print("Start training on GPU...")
#     # print(f"Device: {device}")
#     # print(f"Sequence length T: {args.sequence_length}")
#     # print(f"Save dir: {args.save_dir}")
#     # print(f"Result txt: {result_txt_path}")

#     # 先写入实验配置信息
#     with open(args.save_path, 'w') as f:
#         # f.write("===== Experiment Config =====\n")
#         # for k, v in vars(args).items():
#         #     f.write(f"{k}: {v}\n")
#         # f.write("\n")
#         f.write("===== Per-epoch Metrics =====\n")
#         f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\n")

#     for epoch in range(1, args.epochs + 1):
#         train_loss, train_acc = train_one_epoch(
#             model=model,
#             loader=train_loader,
#             optimizer=optimizer,
#             device=device,
#             sequence_length=args.sequence_length,
#             criterion=criterion,
#         )

#         val_loss, val_acc = evaluate(
#             model=model,
#             loader=val_loader,
#             device=device,
#             sequence_length=args.sequence_length,
#             criterion=criterion,
#         )

#         print(
#             f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
#             f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
#             f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
#         )

#         with open(args.save_path, 'a') as f:
#             f.write(
#                 f"{epoch}\t"
#                 f"{train_loss:.6f}\t"
#                 f"{train_acc:.6f}\t"
#                 f"{val_loss:.6f}\t"
#                 f"{val_acc:.6f}\n"
#             )

#     test_loss, test_acc = evaluate(
#         model=model,
#         loader=test_loader,
#         device=device,
#         sequence_length=args.sequence_length,
#         criterion=criterion,
#     )

#     print(f"Test loss = {test_loss:.4f}, Test acc = {test_acc:.4f}")

#     with open(args.save_path, 'a') as f:
#         f.write("\n===== Final Test Metrics =====\n")
#         f.write(f"test_loss: {test_loss:.6f}\n")
#         f.write(f"test_acc: {test_acc:.6f}\n")