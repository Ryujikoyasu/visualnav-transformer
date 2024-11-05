import os
import wandb
import argparse
import yaml
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad_adapter import NoMaDAdapter
from vint_train.data.twist_dataset_adapter import TwistDataset
from vint_train.training.train_eval_loop_adapter import train_eval_loop_nomad_adapter

def main(config):
    # デバイスの設定
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # データローダーの作成
    train_dataset = TwistDataset(
        data_dir=config["datasets"]["twist_data"]["train"],
        transform=transform,
        context_size=config["context_size"],  # 設定ファイルから値を渡す
        len_traj_pred=config["len_traj_pred"]  # 設定ファイルから値を渡す
    )
    
    train_loader = DataLoader(
        train_dataset,
        
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    
    test_dataset = TwistDataset(
        data_dir=config["datasets"]["twist_data"]["test"],
        transform=transform,
        context_size=config["context_size"],  # 設定ファイルから値を渡す
        len_traj_pred=config["len_traj_pred"]  # 設定ファイルから値を渡す
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # データローダーの作成後に追加
    print("\n=== DataLoader Debug Info ===")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 最初のバッチを取得してみる
    try:
        sample_batch = next(iter(train_loader))
        print("\nSuccessfully loaded first batch:")
        print(f"Batch keys: {sample_batch.keys()}")
        for k, v in sample_batch.items():
            print(f"{k} shape: {v.shape}")
    except Exception as e:
        print("\nError loading first batch:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
    print("========================\n")
    
    # ベースモデルの作成
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    
    base_model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    
    # 事前学習済みの重みをロード
    checkpoint = torch.load(config["pretrained_path"])
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    base_model.load_state_dict(state_dict, strict=False)
    
    # Adapterモデルの作成
    model = NoMaDAdapter(
        base_model=base_model,
        adapter_bottleneck_dim=config["adapter"]["bottleneck_dim"],
        down_dims=config["down_dims"]
    )
    model = model.to(device)
    
    # デバッグ情報の追加
    print("Adapter parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    
    adapter_params = list(model.get_adapter_parameters())
    print(f"Number of adapter parameters: {len(adapter_params)}")
    
    # オプティマイザとスケジューラの設定
    if len(adapter_params) == 0:
        raise ValueError("No adapter parameters found! Check the adapter implementation.")
        
    optimizer = Adam(adapter_params, lr=config["adapter"]["lr"])
    
    # 学習率スケジューラの設定を追加
    lr_scheduler = None
    if config["scheduler"]["name"] == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],
            eta_min=0.0
        )
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # プロジェクトフォルダの設定
    project_folder = os.path.join("experiments", config["project_name"], config["run_name"])
    os.makedirs(project_folder, exist_ok=True)
    
    # 学習の実行
    metrics = train_eval_loop_nomad_adapter(
        model=model,
        train_loader=train_loader,
        test_dataloaders={"test": test_loader},
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        transform=transform,
        device=device,
        train_model=config["train"],
        epochs=config["num_epochs"],
        eval_freq=config["eval_freq"],
        print_log_freq=config["print_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        eval_fraction=config["eval_fraction"],
        goal_mask_prob=config["goal_mask_prob"],
        use_wandb=config["use_wandb"],
        project_folder=project_folder
    )
    
    print(f"Training completed. Final metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/nomad_adapter.yaml")
    args = parser.parse_args()
    
    # 設定の読み込み
    config = OmegaConf.load(args.config)
    
    # 設定の整合性チェックと調整
    if "defaults" in config:
        del config["defaults"]  # defaultsキーを削除（既にマージ済みのため）
    
    # dictに変換
    config = OmegaConf.to_container(config, resolve=True)
    
    # WandBの設定（コメントアウトされたまま）
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            name=f"adapter_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    main(config)