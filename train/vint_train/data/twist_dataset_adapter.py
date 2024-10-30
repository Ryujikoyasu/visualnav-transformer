import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from typing import Tuple, List

class TwistDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # タイムスタンプでソートされたファイルリストを作成
        self.timestamps = []
        for file in os.listdir(data_dir):
            if file.endswith('_image.jpg'):
                timestamp = int(file.split('_')[0])
                self.timestamps.append(timestamp)
        self.timestamps.sort()
        
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        timestamp = self.timestamps[idx]
        
        # 画像の読み込み
        img_path = os.path.join(self.data_dir, 'images', f'{timestamp}.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Twistデータの読み込み
        twist_path = os.path.join(self.data_dir, 'twists', f'{timestamp}.txt')
        with open(twist_path, 'r') as f:
            twist_data = [float(x) for x in f.read().split(',')]
        
        # 必要に応じて画像の前処理
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'twist': torch.tensor(twist_data, dtype=torch.float32)
        } 