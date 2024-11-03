import wandb
import os
import numpy as np
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import time
from models.nomad.adapter_modules import AdapterLayer

from train_utils import train_nomad_adapter, evaluate_nomad_adapter, CustomEMA

def train_eval_loop_nomad_adapter(
    train_model: bool,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    save_freq: int = 10,
):
    """Adapter層を使用したNOMADモデルの学習ループ"""
    latest_path = os.path.join(project_folder, f"latest.pth")
    
    # Adapterのパラメータ名を取得
    adapter_param_names = [n for n, p in model.named_parameters() if 'adapter' in n]
    
    # CustomEMAの初期化
    ema_model = CustomEMA(
        model=model,
        decay=0.75,
        param_names=adapter_param_names
    )

    # 訓練開始時刻を記録
    start_time = time.time()
    best_test_loss = float('inf')
    final_train_loss = None
    final_test_loss = None
    current_train_loss = None  # 現在のエポックの訓練損失を保持
    current_test_loss = None   # 現在のエポックのテスト損失を保持

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(f"Start NoMaD Adapter Training Epoch {epoch}/{current_epoch + epochs - 1}")
            
            # train_nomad_adapter関数を修正して損失値を返すようにする
            current_train_loss = train_nomad_adapter(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                alpha=alpha,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )
            final_train_loss = current_train_loss  # 最新の訓練損失を保存

            # エポックごとの訓練損失を出力
            print(f"Epoch {epoch} - Train Loss: {current_train_loss:.6f}")

        # 指定したエポック間隔でモデルを保存
        if (epoch + 1) % save_freq == 0 or epoch == current_epoch + epochs - 1:  # save_freq間隔または最終エポックで保存
            
            # 元のモデルの保存
            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)

            # EMAモデルの保存
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.state_dict(), numbered_path)
            print(f"Saved EMA model to {numbered_path}")

            # オプティマイザとスケジューラの保存
            optimizer_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
            torch.save(optimizer.state_dict(), optimizer_path)
            
            scheduler_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
            if lr_scheduler:
                torch.save(lr_scheduler.state_dict(), scheduler_path)

            # アダプターのパラメータのみを抽出して保存
            adapter_state_dict = {
                name: param.state_dict() 
                for name, param in model.named_modules() 
                if isinstance(param, AdapterLayer)
            }
            adapter_path = os.path.join(project_folder, f"adapter_{epoch}.pth")
            torch.save(adapter_state_dict, adapter_path)
            print(f"Saved checkpoint at epoch {epoch}")

        # 評価
        if (epoch + 1) % eval_freq == 0:
            for dataset_type, test_loader in test_dataloaders.items():
                print(f"Start {dataset_type} Testing Epoch {epoch}/{current_epoch + epochs - 1}")
                model.eval()
                current_test_loss = evaluate_nomad_adapter(  # 評価関数も損失値を返すように修正
                    eval_type=dataset_type,
                    model=model,
                    ema_model=ema_model,
                    dataloader=test_loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
                final_test_loss = current_test_loss  # 最新のテスト損失を保存
                
                # エポックごとのテスト損失を出力
                print(f"Epoch {epoch} - Test Loss: {current_test_loss:.6f}")
                
                # ベスト損失の更新と出力
                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    print(f"New Best Test Loss: {best_test_loss:.6f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        # エポックの要約を出力
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {current_train_loss:.6f}")
        print(f"Test Loss: {current_test_loss:.6f}")
        print(f"Best Test Loss: {best_test_loss:.6f}\n")

    # 訓練終了時刻を記録
    end_time = time.time()
    training_time = end_time - start_time

    # メトリクスの記録
    metrics = {
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "best_test_loss": best_test_loss,
        "training_time": training_time,
        "total_epochs": epochs,
        "early_stopped": False,
        "best_epoch": epoch,
    }
    
    return metrics