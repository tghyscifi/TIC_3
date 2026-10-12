import os
import sys
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.LeNet.lenet_model.lenet import LeNet


class MNISTJPEGDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        pattern = re.compile(r'^(\d+)_(\d+)\.jpg$')
        for fname in tqdm(os.listdir(root_dir), desc='Loading images', unit='file'):
            m = pattern.match(fname)
            if m is None:
                continue
            label = int(m.group(1))
            self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.root_dir, fname)
        img = Image.open(path).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0) / 255.0
        img_tensor = img_tensor * 2.0 - 1.0
        return img_tensor, label


def dequantize_weights(state_dict):
    for k in list(state_dict.keys()):
        if 'weight' in k:
            state_dict[k] = state_dict[k].float() / 128.0
        elif 'bias' in k:
            state_dict[k] = state_dict[k].float() / (128.0 * 128.0)
    return state_dict


def requantize_weights(state_dict):
    for k in list(state_dict.keys()):
        if 'weight' in k:
            state_dict[k] = (state_dict[k] * 128.0).round().clamp(-128, 127)
        elif 'bias' in k:
            state_dict[k] = (state_dict[k] * 128.0 * 128.0).round().clamp(-0x80000000, 0x7fffffff)
    return state_dict


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_dir = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg'
    print('Scanning dataset...')
    dataset = MNISTJPEGDataset(data_dir)
    print(f'Total samples: {len(dataset)}')

    label_counts = {}
    for _, lbl in dataset.samples:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print('Label distribution:')
    print('  '.join([f'{k}:{v}' for k, v in sorted(label_counts.items())]))

    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    print(f'Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}')

    train_loader = DataLoader(dataset, batch_size=64, sampler=SubsetRandomSampler(train_idx), num_workers=0)
    val_loader = DataLoader(dataset, batch_size=64, sampler=SubsetRandomSampler(val_idx), num_workers=0)

    model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=False, aware=False)

    ckpt_path = 'generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth'
    if os.path.exists(ckpt_path):
        print(f'Loading pretrained checkpoint: {ckpt_path}')
        pretrained_dict = torch.load(ckpt_path, map_location=device)
        if pretrained_dict.get('state_dict') is not None:
            pretrained_dict = pretrained_dict['state_dict']

        pretrained_dict = dequantize_weights(pretrained_dict)
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        if missing:
            print(f'Missing keys: {missing}')
        if unexpected:
            print(f'Unexpected keys: {unexpected}')
        print('Pretrained weights loaded and dequantized.')
    else:
        print('No pretrained checkpoint found, training from scratch.')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0

    print(f'\n{"="*60}')
    print(f'Starting training for {num_epochs} epochs')
    print(f'{"="*60}\n')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch',
                    bar_format='{l_bar}{bar:30}{r_bar}')
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item() / images.size(0):.4f}')

        avg_epoch_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_epoch_loss / 64)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', unit='batch',
                                       bar_format='{l_bar}{bar:20}{r_bar}'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        history['val_acc'].append(acc)

        if acc > best_acc:
            best_acc = acc

        print(f'  -> Loss: {avg_epoch_loss / 64:.4f}  |  Val Acc: {acc:.2f}%  |  Best: {best_acc:.2f}%')
        print()

    state_dict = model.state_dict()
    state_dict = requantize_weights(state_dict)

    save_path = 'generator/LeNet/lenet_model/lenet_truth.pth'
    torch.save(state_dict, save_path)
    print(f'{"="*60}')
    print(f'Training complete!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Final validation accuracy: {history["val_acc"][-1]:.2f}%')
    print(f'Model saved (re-quantized) to {save_path}')
    print(f'{"="*60}')

    print('\nTraining history:')
    print(f'  Epoch |  Loss  | Val Acc')
    print(f'  ------+--------+--------')
    for i in range(num_epochs):
        print(f'  {i+1:5d} | {history["train_loss"][i]:.4f} | {history["val_acc"][i]:.2f}%')


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    train()
