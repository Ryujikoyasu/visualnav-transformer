import torch
from torch.utils.data import Dataset
import os
import cv2
import pickle
import glob
from typing import Dict, List

class TwistDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, context_size: int = 5, len_traj_pred: int = 8):
        """
        Args:
            data_dir: processed_data/train または processed_data/test へのパス
            transform: 画像の前処理
            context_size: コンテキストとして使用する画像数
            len_traj_pred: 予測するTwistの数
        """
        self.transform = transform
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        
        # トラジェクトリディレクトリの一覧を取得
        self.traj_dirs = sorted(glob.glob(os.path.join(data_dir, 'traj_*')))
        if not self.traj_dirs:
            raise ValueError(f"No trajectory directories found in {data_dir}")
        
        print(f"Found {len(self.traj_dirs)} trajectory directories in {data_dir}")
        
        # 各トラジェクトリのデータを読み込む
        self.all_data = []
        for traj_dir in self.traj_dirs:
            # traj_data.pklを読み込む
            with open(os.path.join(traj_dir, 'traj_data.pkl'), 'rb') as f:
                traj_data = pickle.load(f)
            
            # 有効なインデックスの計算
            num_frames = len(traj_data['normalized_twists'])
            valid_indices = []
            for i in range(num_frames - (self.context_size + self.len_traj_pred) + 1):
                valid_indices.append({
                    'traj_dir': traj_dir,
                    'start_idx': i,
                    'traj_data': traj_data
                })
            self.all_data.extend(valid_indices)
        
        print(f"Total number of valid sequences: {len(self.all_data)}")
    
    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_info = self.all_data[idx]
        traj_dir = data_info['traj_dir']
        start_idx = data_info['start_idx']
        traj_data = data_info['traj_data']
        
        # コンテキスト画像の読み込み
        context_images = []
        for i in range(self.context_size):
            img_path = os.path.join(traj_dir, f'{start_idx + i:06d}.jpg')
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            context_images.append(image)
        
        # 画像の形状を変更
        context_images = torch.stack(context_images)  # (context_size, C, H, W)
        context_images = context_images.view(1, -1, context_images.size(2), context_images.size(3))  # (1, context_size*C, H, W)
        
        # ゴール画像の読み込み（データセットの最後の画像）
        goal_img_path = os.path.join(traj_dir, f'{len(traj_data["normalized_twists"]) - 1:06d}.jpg')
        goal_image = cv2.imread(goal_img_path)
        if goal_image is None:
            raise ValueError(f"Failed to load goal image: {goal_img_path}")
        goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            goal_image = self.transform(goal_image)
        goal_image = goal_image.unsqueeze(0)  # (1, C, H, W)
        
        # 予測対象のTwistデータ
        twist_indices = range(start_idx + self.context_size, 
                             start_idx + self.context_size + self.len_traj_pred)
        twist_data = [traj_data['normalized_twists'][i] for i in twist_indices]
        
        return {
            'image': context_images,  # (1, context_size*C, H, W)
            'goal_image': goal_image,  # (1, C, H, W)
            'twist': torch.tensor(twist_data, dtype=torch.float32),  # (len_traj_pred, 2)
        }