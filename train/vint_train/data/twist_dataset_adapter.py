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
        twist_data_list = []
        
        # データの読み込みと統計量の計算
        for file in os.listdir(data_dir):
            if file.endswith('_image.jpg'):
                timestamp = int(file.split('_')[0])
                twist_path = os.path.join(data_dir, f'{timestamp}_twist.txt')
                
                if os.path.exists(twist_path):
                    with open(twist_path, 'r') as f:
                        twist_data = [float(x) for x in f.read().split(',')]
                        twist_data_list.append(twist_data)
                        self.timestamps.append(timestamp)
        
        if not self.timestamps:
            raise ValueError(f"No valid data found in {data_dir}")
            
        # 統計量の計算
        twist_data_array = np.array(twist_data_list)
        self.twist_mean = np.mean(twist_data_array, axis=0)
        self.twist_std = np.std(twist_data_array, axis=0)
        
        # ゼロ除算を防ぐ
        self.twist_std = np.where(self.twist_std == 0, 1.0, self.twist_std)
        
        self.timestamps.sort()
        
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        timestamp = self.timestamps[idx]
        
        # 画像の読み込み
        img_path = os.path.join(self.data_dir, f'{timestamp}_image.jpg')
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Twistデータの読み込み
        twist_path = os.path.join(self.data_dir, f'{timestamp}_twist.txt')
        with open(twist_path, 'r') as f:
            twist_data = np.array([float(x) for x in f.read().split(',')])
        
        # 標準化
        twist_data = (twist_data - self.twist_mean) / self.twist_std
        
        # 必要に応じて画像の前処理
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'twist': torch.tensor(twist_data, dtype=torch.float32)
        }

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