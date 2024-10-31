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
        
        # Twistの統計情報を計算
        all_twists = []
        for traj in self.trajectories:
            traj_path = os.path.join(data_dir, traj)
            with open(os.path.join(traj_path, 'traj_data.pkl'), 'rb') as f:
                traj_data = pickle.load(f)
                all_twists.extend(traj_data['twists'])
        
        # 正規化のための統計量を計算
        all_twists = np.array(all_twists)
        self.twist_mean = np.mean(all_twists, axis=0)
        self.twist_std = np.std(all_twists, axis=0)
        self.twist_std[self.twist_std == 0] = 1.0  # ゼロ除算を防ぐ
        
        # データの読み込みと正規化
        self.data = []
        for traj in self.trajectories:
            traj_path = os.path.join(data_dir, traj)
            with open(os.path.join(traj_path, 'traj_data.pkl'), 'rb') as f:
                traj_data = pickle.load(f)
            
            for i in range(len(traj_data['twists'])):
                # Twistデータを正規化
                normalized_twist = (traj_data['twists'][i] - self.twist_mean) / self.twist_std
                
                self.data.append({
                    'image_path': os.path.join(traj_path, f'{i}.jpg'),
                    'twist': normalized_twist,
                    'twist_raw': traj_data['twists'][i]  # 生のTwistデータも保持
                })
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        # 画像の読み込み
        image = cv2.imread(item['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'twist': torch.tensor(item['twist'], dtype=torch.float32),
            'twist_raw': torch.tensor(item['twist_raw'], dtype=torch.float32)
        }
    
    def denormalize_twist(self, normalized_twist: np.ndarray) -> np.ndarray:
        """正規化されたTwistを元のスケールに戻す"""
        return normalized_twist * self.twist_std + self.twist_mean

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """正規化の統計量を取得"""
        return self.twist_mean, self.twist_std

def format_twist_dataset(source_dir: str = DEFAULT_RAW_DIR, 
                        target_dir: str = DEFAULT_PROCESSED_DIR,
                        train_ratio: float = 0.8):
    """
    ソースデータを必要なフォーマットに変換し、train/testに分割する
    
    Args:
        source_dir (str): 元のデータセットのディレクトリ
        target_dir (str): フォーマット済みデータセットの出力先
        train_ratio (float): 訓練データの割合
    """
    # トレーニングとテスト用のディレクトリを作成
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Twistデータの統計を計算
    twist_files = sorted(os.listdir(os.path.join(source_dir, 'twists')))
    all_twists = []
    
    for twist_file in twist_files:
        with open(os.path.join(source_dir, 'twists', twist_file), 'r') as f:
            twist = [float(x) for x in f.read().split(',')]
            all_twists.append(twist)
    
    # データをシャッフルしてtrain/testに分割
    indices = np.random.permutation(len(twist_files))
    train_size = int(len(indices) * train_ratio)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # データを保存
    def save_data(indices, output_dir):
        traj_data = {'twists': [all_twists[i] for i in indices]}
        traj_path = os.path.join(output_dir, 'trajectory_001')
        os.makedirs(traj_path, exist_ok=True)
        
        # 画像をコピー
        for new_idx, old_idx in enumerate(indices):
            src = os.path.join(source_dir, 'images', twist_files[old_idx].replace('.txt', '.jpg'))
            dst = os.path.join(traj_path, f'{new_idx}.jpg')
            os.system(f'cp {src} {dst}')
        
        # traj_data.pklを保存
        with open(os.path.join(traj_path, 'traj_data.pkl'), 'wb') as f:
            pickle.dump(traj_data, f)
    
    save_data(train_indices, train_dir)
    save_data(test_indices, test_dir)

if __name__ == '__main__':
    # データセットを変換
    format_twist_dataset()
    
    # データセットをロードしてテスト
    dataset = TwistDataset()
    print(f"データセットのサイズ: {len(dataset)}")
    
    # 正規化の統計量を表示
    mean, std = dataset.get_normalization_stats()
    print(f"Twist mean: {mean}")
    print(f"Twist std: {std}")
    
    # サンプルデータの表示
    sample = dataset[0]
    print(f"正規化されたTwist: {sample['twist']}")
    print(f"元のTwist: {sample['twist_raw']}")