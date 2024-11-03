import wandb
import os
import numpy as np
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from diffusers.training_utils import EMAModel
import time

from .train_eval_loop import train_eval_loop_nomad
from ..models.nomad.nomad_adapter import NoMaDAdapter

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
):
    """
    Adapter層を使用したNOMADモデルの学習ループ
    
    Args:
        train_model (bool): 学習を実行るかどうか
        model (torch.nn.Module): NoMaDAdapterモデル
        optimizer (torch.optim.Optimizer): オプティマイザ
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): 学習率スケジューラ
        noise_scheduler: ノイズスケジューラ
        train_loader (DataLoader): 訓練データローダー
        test_dataloaders (Dict[str, DataLoader]): テストデータローダーの辞書
        transform: 画像の前処理
        goal_mask_prob (float): ゴールマスクの確率（NoMaDとの互換性のため）
        epochs (int): 学習エポック数
        device (torch.device): 使用するデバイス
        project_folder (str): モデルと結果を保存するフォルダ
        print_log_freq (int): ログ出力の頻度
        wandb_log_freq (int): wandbログの頻度
        image_log_freq (int): 画像ログの頻度（NoMaDとの互換性のため）
        num_images_log (int): ログする画像数（NoMaDとの互換性のため）
        current_epoch (int): 開始エポック
        alpha (float): 損失の重み（NoMaDとの互換性のため）
        use_wandb (bool): wandbを使用するかどうか
        eval_fraction (float): 評価に使用するデータの割合
        eval_freq (int): 評価の頻度
        
    Returns:
        dict: 訓練の結果を含む辞書
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    
    # Adapterのパラメータを取得
    adapter_params = [p for n, p in model.named_parameters() if 'adapter' in n]
    
    # EMAModelの初期化
    ema_model = EMAModel(
        model=model,
        power=0.75,
        parameters=adapter_params
    )
    
    # 訓練開始時刻を記録
    start_time = time.time()
    best_test_loss = float('inf')
    final_train_loss = None
    final_test_loss = None

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(f"Start NoMaD Adapter Training Epoch {epoch}/{current_epoch + epochs - 1}")
            
            model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # データの準備
                images = batch['image'].to(device)  # (B, context_size*C, H, W)
                goal_images = batch['goal_image'].to(device)  # (B, C, H, W)
                twists = batch['twist'].to(device)  # (B, len_traj_pred, 2)
                
                B = twists.shape[0]  # バッチサイズ
                
                # タイムステップの生成
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()
                
                # ノイズの追加
                noise = torch.randn_like(twists)
                noisy_twists = noise_scheduler.add_noise(twists, noise, timesteps)
                
                # モデルの予測
                noise_pred = model(images, goal_images, noisy_twists, timesteps)
                
                # 損失の計算
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # 逆伝播
                loss.backward()
                optimizer.step()
                ema_model.step(model)
                
                total_loss += loss.item()
                num_batches += 1

                # ログの出力
                if (batch_idx + 1) % print_log_freq == 0:
                    print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
                if use_wandb and (batch_idx + 1) % wandb_log_freq == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/step": batch_idx,
                    })

            # エポックごとの平均損失を更新
            avg_loss = total_loss / num_batches
            final_train_loss = avg_loss

        # モデルの保存
        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # オプティマイザとスケジューラの保存
        optimizer_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        torch.save(optimizer.state_dict(), optimizer_path)
        
        scheduler_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        torch.save(lr_scheduler.state_dict(), scheduler_path)

        # 評価
        if (epoch + 1) % eval_freq == 0:
            for dataset_type, test_loader in test_dataloaders.items():
                print(f"Start {dataset_type} Testing Epoch {epoch}/{current_epoch + epochs - 1}")
                
                model.eval()
                test_loss = 0
                num_test_batches = 0

                with torch.no_grad():
                    for test_batch in test_loader:
                        images = test_batch['image'].to(device)
                        goal_images = test_batch['goal_image'].to(device)
                        twists = test_batch['twist'].to(device)
                        
                        # 評価時もゴール画像を常にマスク
                        goal_mask = torch.ones(images.shape[0], 1, device=device)
                        
                        noise = torch.randn_like(twists)
                        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (twists.shape[0],), device=device).long()
                        noisy_twists = noise_scheduler.add_noise(twists, noise, timesteps)
                        
                        # ゴールマスクを追加
                        noise_pred = ema_model.averaged_model(images, goal_images, noisy_twists, timesteps, goal_mask=goal_mask)
                        
                        # 損失の計算
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)
                        test_loss += loss.item()
                        num_test_batches += 1

                avg_test_loss = test_loss / num_test_batches
                final_test_loss = avg_test_loss
                
                # 最良モデルの更新
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    # 最良モデルを保存
                    best_model_path = os.path.join(project_folder, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema_model.averaged_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                        'loss': best_test_loss,
                    }, best_model_path)
                    print(f"Saved best model with loss {best_test_loss:.4f}")

                if use_wandb:
                    wandb.log({
                        f"test/{dataset_type}_loss": avg_test_loss,
                        "test/epoch": epoch,
                    })

        # 学習率の更新
        if lr_scheduler is not None:
            lr_scheduler.step()
            if use_wandb:
                wandb.log({
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                })

    # 訓練終了時刻を記録
    end_time = time.time()
    training_time = end_time - start_time

    print("Training completed!")

    # 訓練の結果を記録
    metrics = {
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "best_test_loss": best_test_loss,
        "training_time": training_time,
        "total_epochs": epochs,
        "early_stopped": False,  # 早期停止を実装する場合はここを更新
        "best_epoch": epoch,
    }
    
    # メトリクスの出力
    print("\nTraining Summary:")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Best test loss: {best_test_loss:.4f}")
    
    return metrics
