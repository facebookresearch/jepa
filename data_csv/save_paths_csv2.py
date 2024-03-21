import glob
import csv
import random
import os

# 定义目标文件夹路径
directory_path = '/beacon/data01/chengjie.zheng001/data/kinetics-dataset/k700-2020/train/'

# 定义支持的文件扩展名
extensions = ['mp4']

# 查找目录下所有匹配的文件
files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(directory_path, f'**/*.{ext}'), recursive=True))

# 生成随机整数标签
labels = [random.randint(0, 100) for _ in files]

# CSV 文件保存路径
csv_file_path = '/beacon/data01/chengjie.zheng001/Projects/MGH/umb-jepa/data_csv/k700_train.csv'

# 保存到 CSV 文件
with open(csv_file_path, 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile)
    for file_path, label in zip(files, labels):
        formatted_string = f"{file_path} ${label}"
        filewriter.writerow([formatted_string])

print(f"Saved {len(files)} entries to {csv_file_path}")