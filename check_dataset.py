import os
import cv2
import numpy as np

def check_dataset(data_dir):
    # ファイルの存在確認
    image_files = [f for f in os.listdir(data_dir) if f.endswith('_image.jpg')]
    twist_files = [f for f in os.listdir(data_dir) if f.endswith('_twist.txt')]
    
    print(f"画像ファイル数: {len(image_files)}")
    print(f"Twistファイル数: {len(twist_files)}")
    
    # サンプルデータの確認
    if image_files:
        # 画像の確認
        sample_image = cv2.imread(os.path.join(data_dir, image_files[0]))
        print(f"画像サイズ: {sample_image.shape}")
        
        # Twistデータの確認
        sample_twist = image_files[0].replace('_image.jpg', '_twist.txt')
        with open(os.path.join(data_dir, sample_twist), 'r') as f:
            twist_data = f.read().strip().split(',')
            print(f"Twistデータ: {twist_data}")

if __name__ == '__main__':
    check_dataset('/path/to/save/directory') 