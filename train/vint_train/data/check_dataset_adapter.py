import os
import glob
import numpy as np
import pickle
from typing import Dict, Tuple, List

def process_dataset(data_dir: str, output_dir: str) -> str:
    """
    生データを処理して単一のデータセットを作成する
    
    Args:
        data_dir: 生データのディレクトリ（traj_YYYYMMDD_HHMMSS形式）
        output_dir: 処理済みデータの出力先ディレクトリ
    
    Returns:
        str: 処理済みデータのディレクトリパス
    """
    # 入力データの確認
    image_files = sorted(glob.glob(os.path.join(data_dir, 'images', '*.jpg')))
    twist_files = sorted(glob.glob(os.path.join(data_dir, 'twists', '*.txt')))
    
    print(f"=== Dataset Processing Info ===")
    print(f"Source directory: {data_dir}")
    print(f"Number of images: {len(image_files)}")
    print(f"Number of twists: {len(twist_files)}")
    
    # Twistデータの読み込みと正規化
    twists = []
    for twist_file in twist_files:
        with open(twist_file, 'r') as f:
            twist_data = [float(x) for x in f.read().strip().split(',')]
            twists.append(twist_data)
    
    twists = np.array(twists)
    twist_mean = np.mean(twists, axis=0)
    twist_std = np.std(twists, axis=0)
    twist_std = np.where(twist_std == 0, 1.0, twist_std)
    
    # 正規化されたTwistデータ
    normalized_twists = (twists - twist_mean) / twist_std
    
    # 出力ディレクトリの作成
    traj_name = os.path.basename(data_dir)
    target_dir = os.path.join(output_dir, traj_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # データの保存
    for i, (src_img, norm_twist) in enumerate(zip(image_files, normalized_twists)):
        # 画像のシンボリックリンク作成
        dst_img = os.path.join(target_dir, f'{i:06d}.jpg')
        if os.path.exists(dst_img):
            os.remove(dst_img)
        os.symlink(src_img, dst_img)
    
    # traj_data.pklの保存（正規化前後のデータと統計量を含む）
    traj_data = {
        'raw_twists': twists.tolist(),
        'normalized_twists': normalized_twists.tolist(),
        'twist_mean': twist_mean.tolist(),
        'twist_std': twist_std.tolist()
    }
    pkl_path = os.path.join(target_dir, 'traj_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(traj_data, f)
    
    print(f"\nDataset processing completed!")
    print(f"- Images: {target_dir}")
    print(f"- Trajectory data: {pkl_path}")
    print(f"- Total frames: {len(image_files)}")
    print("========================")
    
    return target_dir

def process_all_trajectories(raw_data_dir: str, output_dir: str) -> List[str]:
    """
    raw_dataディレクトリ内の全てのトラジェクトリデータを処理する
    
    Args:
        raw_data_dir: 生データのルートディレクトリ
        output_dir: 処理済みデータの出力先ディレクトリ
    
    Returns:
        List[str]: 処理されたデータセットのディレクトリパスのリスト
    """
    # traj_で始まるディレクトリを全て取得
    traj_dirs = sorted(glob.glob(os.path.join(raw_data_dir, 'traj_*')))
    
    if not traj_dirs:
        raise ValueError(f"No trajectory directories found in {raw_data_dir}")
    
    print(f"Found {len(traj_dirs)} trajectory directories")
    processed_dirs = []
    
    for traj_dir in traj_dirs:
        print(f"\nProcessing trajectory: {os.path.basename(traj_dir)}")
        try:
            processed_dir = process_dataset(traj_dir, output_dir)
            processed_dirs.append(processed_dir)
        except Exception as e:
            print(f"Error processing {traj_dir}: {str(e)}")
            continue
    
    print(f"\nProcessed {len(processed_dirs)} trajectories successfully")
    return processed_dirs

if __name__ == "__main__":
    raw_data_dir = "/ssd/source/navigation/asset/nomad_adapter_dataset/raw_data"
    output_dir = "/ssd/source/navigation/asset/nomad_adapter_dataset/processed_data"
    
    processed_dirs = process_all_trajectories(raw_data_dir, output_dir)
    
    print("\nProcessed directories:")
    for dir_path in processed_dirs:
        print(f"- {dir_path}")