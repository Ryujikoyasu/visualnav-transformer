import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import pickle
from typing import Tuple, Dict

class TwistDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, context_size: int = 5, len_traj_pred: int = 8):
        """
        Args:
            data_dir: processed_data/train または processed_data/test へのパス
            transform: 画像の前処理
            context_size: コンテキストサイズ
            len_traj_pred: 軌道予測の長さ
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # trajectory_001ディレクトリのパス
        traj_dir = os.path.join(data_dir, 'trajectory_001')
        
        # traj_data.pklを読み込む
        with open(os.path.join(traj_dir, 'traj_data.pkl'), 'rb') as f:
            self.traj_data = pickle.load(f)
        
        # 画像ファイルの数を確認
        self.num_frames = len([f for f in os.listdir(traj_dir) if f.endswith('.jpg')])
        
        if self.num_frames == 0:
            raise ValueError(f"No images found in {traj_dir}")
        
        # Twistデータの統計量を計算
        twist_data = np.array(self.traj_data['twists'])
        self.twist_mean = np.mean(twist_data, axis=0)
        self.twist_std = np.std(twist_data, axis=0)
        self.twist_std = np.where(self.twist_std == 0, 1.0, self.twist_std)
        
        # デバッグ用の情報出力
        print(f"=== Dataset Debug Info ===")
        print(f"Data directory: {data_dir}")
        print(f"Number of image files: {self.num_frames}")
        print(f"Number of twist data: {len(self.traj_data['twists'])}")
        print(f"Context size: {context_size}")
        print(f"Trajectory prediction length: {len_traj_pred}")
        
        # valid_indicesの計算過程を表示
        self.valid_indices = []
        for i in range(self.num_frames - (self.context_size + self.len_traj_pred) + 1):
            if i + self.context_size + self.len_traj_pred <= len(self.traj_data['twists']):
                self.valid_indices.append(i)
                
        print(f"Number of valid indices: {len(self.valid_indices)}")
        if len(self.valid_indices) > 0:
            print(f"First few valid indices: {self.valid_indices[:5]}")
        print("========================")
    
    def __len__(self) -> int:
        return self.num_frames
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # デバッグ用の情報出力
        print(f"\n=== GetItem Debug Info ===")
        print(f"Requested index: {idx}")
        print(f"Valid indices length: {len(self.valid_indices)}")
        
        start_idx = self.valid_indices[idx]
        print(f"Start index: {start_idx}")
        print(f"Context range: {start_idx} to {start_idx + self.context_size - 1}")
        print(f"Twist range: {start_idx + self.context_size} to {start_idx + self.context_size + self.len_traj_pred - 1}")
        print("========================\n")
        
        # 画像の読み込み
        img_path = os.path.join(self.data_dir, 'trajectory_001', f'{start_idx}.jpg')
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 画像の前処理
        if self.transform:
            image = self.transform(image)
        
        # Twistデータの取得と正規化
        twist_data = np.array(self.traj_data['twists'][start_idx])
        twist_data = (twist_data - self.twist_mean) / self.twist_std
        
        # ViNT_Datasetと同じ形式で返す
        return {
            'context': image,                     # 現在の画像
            'goal': image.clone(),                # Twist制御では不要だがダミーとして
            'waypoints': torch.zeros(5, 2),       # ダミーのウェイポイント
            'actions': torch.tensor(twist_data, dtype=torch.float32),  # Twistデータ
            'distances': torch.zeros(1)           # ダミーの距離
        }
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """正規化のための統計量を返す"""
        return self.twist_mean, self.twist_std
    
    def denormalize_twist(self, twist: torch.Tensor) -> torch.Tensor:
        """正規化されたTwistデータを元のスケールに戻す"""
        twist_np = twist.cpu().numpy()
        denorm_twist = twist_np * self.twist_std + self.twist_mean
        return torch.from_numpy(denorm_twist).to(twist.device)
    
    def get_raw_twist(self, idx: int) -> np.ndarray:
        """指定されたインデックスの生のTwistデータを返す"""
        return np.array(self.traj_data['twists'][idx])
    
    def get_image_path(self, idx: int) -> str:
        """指定されたインデックスの画像パスを返す"""
        return os.path.join(self.data_dir, 'trajectory_001', f'{idx}.jpg')
    
    @property
    def total_frames(self) -> int:
        """総フレーム数を返す（__len__のエイリアス）"""
        return self.num_frames
    
    def get_trajectory_length(self) -> int:
        """軌道の長さを返す"""
        return len(self.traj_data['twists'])
    
    def get_metadata(self) -> Dict:
        """データセットのメタデータを返す"""
        return {
            'num_frames': self.num_frames,
            'twist_mean': self.twist_mean.tolist(),
            'twist_std': self.twist_std.tolist(),
            'data_dir': self.data_dir
        }

def format_twist_dataset(source_dir: str, target_dir: str, train_ratio: float = 0.8):
    """
    ソースデータを必要なフォーマットに変換し、train/testに分割する
    
    Args:
        source_dir: raw_data_4.0hz へのパス
        target_dir: processed_data へのパス
        train_ratio: 訓練データの割合
    """
    # 出力ディレクトリの作成
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    os.makedirs(os.path.join(train_dir, 'trajectory_001'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'trajectory_001'), exist_ok=True)
    
    # 画像とTwistデータの読み込み
    image_files = sorted(os.listdir(os.path.join(source_dir, 'images')))
    twist_files = sorted(os.listdir(os.path.join(source_dir, 'twists')))
    
    # データの分割
    num_samples = len(image_files)
    indices = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # データの保存
    def save_split(indices, output_dir):
        twists = []
        for i, idx in enumerate(indices):
            # 画像のコピー
            src_img = os.path.join(source_dir, 'images', image_files[idx])
            dst_img = os.path.join(output_dir, 'trajectory_001', f'{i}.jpg')
            os.system(f'cp {src_img} {dst_img}')
            
            # Twistデータの読み込み
            with open(os.path.join(source_dir, 'twists', twist_files[idx]), 'r') as f:
                twist = [float(x) for x in f.read().split(',')]
                twists.append(twist)
        
        # traj_data.pklの保存
        traj_data = {'twists': twists}
        with open(os.path.join(output_dir, 'trajectory_001', 'traj_data.pkl'), 'wb') as f:
            pickle.dump(traj_data, f)
    
    save_split(train_indices, train_dir)
    save_split(test_indices, test_dir)