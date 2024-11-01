import torch
from torch.utils.data import Dataset
import os
import cv2
import pickle
from typing import Dict

class TwistDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, context_size: int = 5, len_traj_pred: int = 8):
        """
        Args:
            data_dir: 処理済みデータのディレクトリパス
            transform: 画像の前処理
            context_size: コンテキストとして使用する画像数
            len_traj_pred: 予測するTwistの数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        
        # traj_data.pklを読み込む
        with open(os.path.join(data_dir, 'traj_data.pkl'), 'rb') as f:
            self.traj_data = pickle.load(f)
        
        # 有効なインデックスの計算
        self.num_frames = len(self.traj_data['normalized_twists'])
        self.valid_indices = []
        for i in range(self.num_frames - (self.context_size + self.len_traj_pred) + 1):
            self.valid_indices.append(i)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.valid_indices[idx]
        
        # コンテキスト画像の読み込み
        context_images = []
        for i in range(self.context_size):
            img_path = os.path.join(self.data_dir, f'{start_idx + i:06d}.jpg')
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            context_images.append(image)
        
        # 予測対象のTwistデータ
        twist_indices = range(start_idx + self.context_size, 
                            start_idx + self.context_size + self.len_traj_pred)
        twist_data = [self.traj_data['normalized_twists'][i] for i in twist_indices]
        
        return {
            'image': torch.stack(context_images),  # (context_size, C, H, W)
            'twist': torch.tensor(twist_data, dtype=torch.float32),  # (len_traj_pred, 2)
        }