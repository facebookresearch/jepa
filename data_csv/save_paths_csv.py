import glob
import pandas as pd
import random
import os

# 指定目录路径
directory_path = '/beacon/data01/chengjie.zheng001/00_datasets/kinetics-dataset/k700-2020/val'

# 使用 glob 查找所有的 mp4 文件
mp4_files = glob.glob(os.path.join(directory_path, '**/*.mp4'), recursive=True)

# 生成与文件列表相同长度的随机整数列表
random_integers = '$'+str([random.randint(0, 700) for _ in mp4_files])
# _ 代表的是对 mp4_files 列表中的每个元素进行迭代，但实际上我们不需要在迭代过程中使用这些元素的值。目的仅仅是为了确保生成与 mp4_files 列表长度相同的随机整数列表。
# 使用 _ 作为变量名称是一种约定，表明该变量是暂时或不被使用的，这有助于提高代码的可读性，让读者知道这个变量在循环体内部没有被用到

# 创建一个 DataFrame
df = pd.DataFrame({
    'FilePath': mp4_files,
    'RandomInteger': random_integers
})

# 保存到 CSV 文件
csv_file_path = '/beacon/data01/chengjie.zheng001/code/MGH/jepa/data_csv/k700_val.csv'
df.to_csv(csv_file_path, index=False)