import os


def rename_files_in_directory(directory):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否以 "_0.png" 结尾
        # if filename.endswith("_0.json"):
        if filename.startswith("color"):
            # 构造新的文件名，去掉 "_0"
            new_filename = filename.replace("color", "7_28")

            # 获取旧文件路径和新文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)


            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")


# 使用方法
directory_path = r'D:\python-ai\test\tomato_data_cut_10_5\obzg_sdk\7_28_data'  # 修改为你的文件夹路径
rename_files_in_directory(directory_path)
