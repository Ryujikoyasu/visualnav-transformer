import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import pickle
from typing import Tuple, List

# デフォルトのディレクトリパスを設定
DEFAULT_DATA_ROOT = "/ssd/source/navigation/asset/nomad_adapter_dataset"
DEFAULT_RAW_DIR = os.path.join(DEFAULT_DATA_ROOT, 'raw_data_4.0hz')
DEFAULT_PROCESSED_DIR = os.path.join(DEFAULT_DATA_ROOT, 'processed_data')

class TwistDataset(Dataset):
    def __init__(self, data_dir: str = DEFAULT_PROCESSED_DIR, transform=None):
        """
        Args:
            data_dir (str): データセットのルートディレクトリ
            transform: 画像の前処理
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # トラジェクトリのリストを取得
        self.trajectories = []
        for traj_dir in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, traj_dir)):
                self.trajectories.append(traj_dir)
        
        # 各トラジェクトリの情報を読み込む
        self.data = []
        for traj in self.trajectories:
            traj_path = os.path.join(data_dir, traj)
            
            # traj_data.pklを読み込む
            with open(os.path.join(traj_path, 'traj_data.pkl'), 'rb') as f:
                traj_data = pickle.load(f)
            
            # 画像とtwistのペアを保存
            for i in range(len(traj_data['twists'])):
                self.data.append({
                    'image_path': os.path.join(traj_path, f'{i}.jpg'),
                    'twist': traj_data['twists'][i],
                })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        # 画像の読み込み
        image = cv2.imread(item['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 必要に応じて画像の前処理
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'twist': torch.tensor(item['twist'], dtype=torch.float32)
        }

def format_twist_dataset(source_dir: str = DEFAULT_RAW_DIR, 
                        target_dir: str = DEFAULT_PROCESSED_DIR):
    """
    ソースデータを必要なフォーマットに変換する
    
    Args:
        source_dir (str): 元のデータセットのディレクトリ
        target_dir (str): フォーマット済みデータセットの出力先
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # 画像とtwistのデータを読み込む
    image_dir = os.path.join(source_dir, 'images')
    twist_dir = os.path.join(source_dir, 'twists')
    
    # トラジェクトリごとにデータを整理
    traj_data = {
        'twists': []
    }
    
    # twistデータを読み込む
    twist_files = sorted(os.listdir(twist_dir))
    for twist_file in twist_files:
        with open(os.path.join(twist_dir, twist_file), 'r') as f:
            twist = [float(x) for x in f.read().split(',')]
            traj_data['twists'].append(twist)
    
    # 新しいフォーマットで保存
    traj_path = os.path.join(target_dir, 'trajectory_001')
    os.makedirs(traj_path, exist_ok=True)
    
    # 画像をコピー
    for i, img_file in enumerate(sorted(os.listdir(image_dir))):
        src = os.path.join(image_dir, img_file)
        dst = os.path.join(traj_path, f'{i}.jpg')
        os.system(f'cp {src} {dst}')
    
    # traj_data.pklを保存
    with open(os.path.join(traj_path, 'traj_data.pkl'), 'wb') as f:
        pickle.dump(traj_data, f)

if __name__ == '__main__':
    # デフォルトのディレクトリを使用してデータセットを変換
    format_twist_dataset()
    
    # データセットの作成
    dataset = TwistDataset()
    print(f"データセットのサイズ: {len(dataset)}")