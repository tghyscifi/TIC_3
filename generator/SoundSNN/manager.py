import os
from multiprocessing import Process


def exe(cmd):
    os.system(cmd)


def run_on_gpu(
    gpu_id: int,
    data_root: str,
    base_save_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    val_size: int,
    seed: int,
    sequence_length: int,
    thresh: float,
):
    # save-dir 里写出当前 thresh 的值
    save_path = os.path.join(base_save_dir, f"thresh_{thresh}.txt")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python sound_snn.py "
        f"--data-root {data_root} "
        f"--save-path {save_path} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--lr {lr} "
        f"--weight-decay {weight_decay} "
        f"--num-workers {num_workers} "
        f"--val-size {val_size} "
        f"--seed {seed} "
        f"--sequence-length {sequence_length} "
        f"--fc1-thresh {thresh} "
        f"--fc1-decay 255 "
        f"--fc1-beta 0 "
        f"--fc2-thresh {thresh} "
        f"--fc2-decay 255 "
        f"--fc2-beta 0 "
        f"--fc3-thresh {thresh} "
        f"--fc3-decay 255 "
        f"--fc3-beta 0 "
        f"--fc4-thresh {thresh} "
        f"--fc4-decay 255 "
        f"--fc4-beta 0"
    )

    print(f"[GPU {gpu_id}] Running: {cmd}")
    exe(cmd)


if __name__ == "__main__":
    # 你可用的 GPU
    gpu_list = [8, 9]

    # 固定参数
    data_root = "/your/path/to/MNIST"
    base_save_dir = "./results"
    epochs = 10
    batch_size = 128
    lr = 1e-3
    weight_decay = 0.0
    num_workers = 16
    val_size = 5000
    seed = 42
    sequence_length = 10

    # 你要搜索的 thresh
    all_thresh = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    processes = []
    num_gpus = len(gpu_list)

    for i, thresh in enumerate(all_thresh):
        gpu_id = gpu_list[i % num_gpus]

        p = Process(
            target=run_on_gpu,
            args=(
                gpu_id,
                data_root,
                base_save_dir,
                epochs,
                batch_size,
                lr,
                weight_decay,
                num_workers,
                val_size,
                seed,
                sequence_length,
                thresh,
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished.")