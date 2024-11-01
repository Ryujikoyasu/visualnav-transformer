import os
import glob
import numpy as np
from datetime import datetime
import pickle

def downsample_dataset(data_dir, target_hz=4.0):
    """データセットを指定のHzにダウンサンプリングする"""
    
    # 画像とtwistのファイル一覧を取得
    image_files = sorted(glob.glob(os.path.join(data_dir, 'images', '*.jpg')))
    twist_files = sorted(glob.glob(os.path.join(data_dir, 'twists', '*.txt')))
    
    # ファイル数の一致を確認
    print(f"元の画像ファイル数: {len(image_files)}")
    print(f"元のTwistファイル数: {len(twist_files)}")
    
    # 少ない方に合わせる
    min_files = min(len(image_files), len(twist_files))
    image_files = image_files[:min_files]
    twist_files = twist_files[:min_files]
    
    # タイムスタンプを抽出して時系列順にソート
    timestamps = []
    paired_files = []  # 画像とTwistのペアを保持
    
    for img_file, twist_file in zip(image_files, twist_files):
        timestamp_str = os.path.basename(img_file).split('.')[0]
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')
            timestamps.append(timestamp)
            paired_files.append((img_file, twist_file))
        except ValueError as e:
            print(f"Warning: Skipping invalid timestamp in file {img_file}")
            continue
    
    # タイムスタンプでソート
    sorted_pairs = [x for _, x in sorted(zip(timestamps, paired_files))]
    
    # 目標の時間間隔（秒）
    target_interval = 1.0 / target_hz
    
    # 保持するペアを選択
    selected_pairs = [sorted_pairs[0]]  # 最初のペアは必ず含める
    last_selected_time = timestamps[0]
    
    for i, ((img_file, twist_file), timestamp) in enumerate(zip(sorted_pairs[1:], timestamps[1:]), 1):
        time_diff = (timestamp - last_selected_time).total_seconds()
        if time_diff >= target_interval:
            selected_pairs.append((img_file, twist_file))
            last_selected_time = timestamp
    
    print(f"ダウンサンプリング後のデータ数: {len(selected_pairs)}")
    
    # 新しいディレクトリを作成
    output_dir = data_dir + f"_{target_hz}hz"
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'twists'), exist_ok=True)
    
    # 選択されたファイルをコピー
    import shutil
    twists = []  # Twistデータを保存するリスト
    
    for i, (src_img, src_twist) in enumerate(selected_pairs):
        # 画像のコピー
        dst_img = os.path.join(output_dir, 'images', f'{i:06d}.jpg')
        shutil.copy2(src_img, dst_img)
        
        # Twistデータの読み込みとコピー
        with open(src_twist, 'r') as f:
            twist_data = [float(x) for x in f.read().strip().split(',')]
            twists.append(twist_data)
        
        dst_twist = os.path.join(output_dir, 'twists', f'{i:06d}.txt')
        shutil.copy2(src_twist, dst_twist)
    
    # traj_data.pklの作成と保存
    traj_data = {'twists': twists}
    with open(os.path.join(output_dir, 'trajectory_001', 'traj_data.pkl'), 'wb') as f:
        pickle.dump(traj_data, f)
    
    print(f"処理完了: {output_dir}")
    return output_dir

if __name__ == "__main__":
    data_dir = "/ssd/source/navigation/asset/nomad_adapter_dataset/raw_data"
    downsample_dataset(data_dir, target_hz=4.0)