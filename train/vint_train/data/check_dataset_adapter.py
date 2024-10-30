import os
import glob
import numpy as np
from datetime import datetime

def downsample_dataset(data_dir, target_hz=4.0):
    """データセットを指定のHzにダウンサンプリングする"""
    
    # 画像とtwistのファイル一覧を取得
    image_files = sorted(glob.glob(os.path.join(data_dir, 'images', '*.jpg')))
    twist_files = sorted(glob.glob(os.path.join(data_dir, 'twists', '*.txt')))
    
    # タイムスタンプを抽出して時系列順にソート
    timestamps = []
    for img_file in image_files:
        timestamp_str = os.path.basename(img_file).split('.')[0]
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
        timestamps.append(timestamp)
    
    # 目標の時間間隔（秒）
    target_interval = 1.0 / target_hz
    
    # 保持するインデックスを選択
    selected_indices = []
    last_selected_time = timestamps[0]
    selected_indices.append(0)
    
    for i, timestamp in enumerate(timestamps[1:], 1):
        time_diff = (timestamp - last_selected_time).total_seconds()
        if time_diff >= target_interval:
            selected_indices.append(i)
            last_selected_time = timestamp
    
    print(f"元のデータ数: {len(image_files)}")
    print(f"ダウンサンプリング後のデータ数: {len(selected_indices)}")
    
    # 新しいディレクトリを作成
    output_dir = data_dir + f"_{target_hz}hz"
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'twists'), exist_ok=True)
    
    # 選択されたファイルをコピー
    import shutil
    for idx in selected_indices:
        # 画像のコピー
        src_img = image_files[idx]
        dst_img = os.path.join(output_dir, 'images', os.path.basename(src_img))
        shutil.copy2(src_img, dst_img)
        
        # Twistデータのコピー
        src_twist = twist_files[idx]
        dst_twist = os.path.join(output_dir, 'twists', os.path.basename(src_twist))
        shutil.copy2(src_twist, dst_twist)
    
    return output_dir

if __name__ == "__main__":
    data_dir = "/path/to/your/raw_data"  # データセットのパスを指定
    downsample_dataset(data_dir, target_hz=4.0)