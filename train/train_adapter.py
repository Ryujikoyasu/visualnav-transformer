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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # データローダーの作成
    train_dataset = TwistDataset(
        data_dir=config["datasets"]["twist_data"]["train"],
        transform=transform
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
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # ベースモデルの作成
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=5,
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    # 位置エンコーディングのバッファを事前に初期化
    vision_encoder.positional_encoding.register_buffer(
        'pos_enc',
        vision_encoder.positional_encoding.pos_enc[:, :5]
    )
    
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
    print("Checkpoint keys:", checkpoint.keys())  # デバッグ用：利用可能なキーを確認
    
    # チェックポイントの構造に応じて適切にロード
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # チェックポイントそのものがstate_dictの場合
        state_dict = checkpoint
    
    # 互換性のないキーをスキップしてロード
    missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    
    print(f"Loaded pretrained model from {config['pretrained_path']}")
    
    # Adapterモデルの作成
    model = NoMaDAdapter(
        base_model=base_model,
        adapter_bottleneck_dim=config["adapter"]["bottleneck_dim"]
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
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # 学習の実行
    metrics = train_eval_loop_nomad_adapter(
        model=model,
        train_loader=train_loader,
        test_dataloaders={"test": test_loader},
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        transform=transform,
        device=device,
        train_model=True,
        epochs=config["num_epochs"],
        eval_freq=config["eval_freq"],
        print_freq=config["print_log_freq"],
        goal_mask_prob=config["goal_mask_prob"]
    )
    
    print(f"Training completed. Final metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/nomad_adapter.yaml")
    args = parser.parse_args()
    
    # まずbase_configを読み込む
    base_config = OmegaConf.load("/ssd/source/navigation/visualnav-transformer/train/config/nomad_adapter.yaml")
    
    # adapter_configを読み込む
    adapter_config = OmegaConf.load(args.config)
    
    # マージする際に、adapter_configを優先させる
    config = OmegaConf.merge(base_config, adapter_config)
    
    # 設定の整合性チェックと調整
    if "defaults" in config:
        del config["defaults"]  # defaultsキーを削除（既にマージ済みのため）
    
    # バッチサイズなどの重要なパラメータが正しく上書きされていることを確認
    print("\nKey Configuration Values:")
    important_keys = [
        "batch_size", "num_workers", "lr", "optimizer", 
        "num_epochs", "eval_freq", "model_type"
    ]
    for key in important_keys:
        print(f"{key}: {config.get(key)}")
    
    print("\nFull Merged Config:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # dictに変換
    config = OmegaConf.to_container(config, resolve=True)
    
    # WandBの設定（コメントアウトされたまま）
    # if config["use_wandb"]:
    #     wandb.login()
    #     wandb.init(
    #         project=config["project_name"],
    #         name=f"adapter_{time.strftime('%Y%m%d_%H%M%S')}",
    #         config=config
    #     )
    
    main(config)