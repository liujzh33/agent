# import os
# import json
# import numpy as np
# from pathlib import Path
# from datetime import datetime

# def analyze_dataset_structure(dataset_path="./btc_2025_06_07"):
#     dataset_path = Path(dataset_path)

#     print("=" * 80)
#     print(f"BTC数据集结构分析 - {dataset_path}")
#     print("=" * 80)

#     if not dataset_path.exists():
#         print(f"❌ 错误: 路径 {dataset_path} 不存在")
#         return

#     structure_output = ""
#     print("\n📁 目录结构:")
#     print("-" * 50)

#     total_dirs = 0
#     total_files = 0
#     file_types = {}
#     file_sizes = []

#     def print_tree(path, prefix="", is_last=True):
#         nonlocal total_dirs, total_files, structure_output

#         items = sorted(list(path.iterdir()))
#         dirs = [item for item in items if item.is_dir()]
#         files = [item for item in items if item.is_file()]

#         for i, dir_path in enumerate(dirs):
#             is_last_dir = (i == len(dirs) - 1) and len(files) == 0
#             connector = "└── " if is_last_dir else "├── "
#             line = f"{prefix}{connector}📁 {dir_path.name}/"
#             print(line)
#             structure_output += line + "\n"
#             total_dirs += 1

#             if len(prefix) < 40:
#                 extension = "    " if is_last_dir else "│   "
#                 print_tree(dir_path, prefix + extension, is_last_dir)

#         for i, file_path in enumerate(files):
#             is_last_file = i == len(files) - 1
#             connector = "└── " if is_last_file else "├── "

#             file_size = file_path.stat().st_size
#             file_sizes.append(file_size)
#             total_files += 1

#             ext = file_path.suffix.lower()
#             file_types[ext] = file_types.get(ext, 0) + 1

#             size_str = format_file_size(file_size)
#             line = f"{prefix}{connector}📄 {file_path.name} ({size_str})"
#             print(line)
#             structure_output += line + "\n"

#     print_tree(dataset_path)

#     print(f"\n📊 数据集概览:")
#     print("-" * 50)
#     print(f"总目录数: {total_dirs}")
#     print(f"总文件数: {total_files}")

#     if file_sizes:
#         total_size = sum(file_sizes)
#         print(f"总大小: {format_file_size(total_size)}")
#         print(f"平均文件大小: {format_file_size(total_size / len(file_sizes))}")

#     print(f"\n📋 文件类型分布:")
#     for ext, count in sorted(file_types.items()):
#         ext_display = ext if ext else "无扩展名"
#         print(f"  {ext_display}: {count} 个文件")

#     generate_dataset_report(dataset_path, total_dirs, total_files, file_types, file_sizes, structure_output)


# def format_file_size(size_bytes):
#     if size_bytes == 0:
#         return "0B"
#     size_names = ["B", "KB", "MB", "GB", "TB"]
#     i = int(np.floor(np.log(size_bytes) / np.log(1024)))
#     p = pow(1024, i)
#     s = round(size_bytes / p, 2)
#     return f"{s} {size_names[i]}"


# def generate_dataset_report(dataset_path, total_dirs, total_files, file_types, file_sizes, structure_output):
#     print(f"\n📋 数据集结构报告生成中...")
#     print("=" * 50)

#     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     txt_content = f"""BTC数据集结构分析报告
# {"=" * 80}
# 分析时间: {current_time}
# 数据集路径: {dataset_path}

# 目录结构统计:
# {"=" * 50}
# 总目录数: {total_dirs}
# 总文件数: {total_files}
# 总大小: {format_file_size(sum(file_sizes)) if file_sizes else '0B'}
# 平均文件大小: {format_file_size(sum(file_sizes)/len(file_sizes)) if file_sizes else '0B'}

# 文件类型分布:
# {"-" * 30}
# """
#     for ext, count in sorted(file_types.items()):
#         ext_display = ext if ext else "无扩展名"
#         txt_content += f"{ext_display}: {count} 个文件\n"

#     txt_content += f"""
# 详细目录结构:
# {"=" * 50}
# {structure_output}
# """

#     # 保存 TXT
#     txt_path = Path.cwd() / f"dataset_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
#     try:
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(txt_content)
#         print("\n📁 TXT 报告已保存至以下路径:")
#         print("=" * 60)
#         print(txt_path)
#         print("=" * 60)
#     except Exception as e:
#         print(f"❌ 保存 TXT 报告失败: {e}")

#     # 保存 JSON
#     json_report = {
#         "dataset_path": str(dataset_path),
#         "analysis_time": datetime.now().isoformat(),
#         "summary": {
#             "total_directories": total_dirs,
#             "total_files": total_files,
#             "total_size": format_file_size(sum(file_sizes)) if file_sizes else '0B',
#             "average_file_size": format_file_size(sum(file_sizes)/len(file_sizes)) if file_sizes else '0B',
#             "file_types": file_types
#         },
#         "directory_structure": structure_output.splitlines()
#     }

#     json_path = Path.cwd() / f"dataset_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     try:
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(json_report, f, indent=2, ensure_ascii=False)
#         print("\n📁 JSON 报告已保存至以下路径:")
#         print("=" * 60)
#         print(json_path)
#         print("=" * 60)
#     except Exception as e:
#         print(f"❌ 保存 JSON 报告失败: {e}")


# if __name__ == "__main__":
#     analyze_dataset_structure("./data/btc_2025_06_07")

#     print("\n📌 使用提示：")
#     print("# 本工具专注于了解目录结构、数据组织方式和文件分布")
#     print("# 可用于初步审查数据集内容、构建索引、数据清洗前检查等场景")


import pandas as pd

csv_path = "data/btc_2025_06_07/spot/trades/data/spot/monthly/trades/BTCUSDT/2025-06-01_2025-07-31/BTCUSDT-trades-2025-06.csv"

try:
    df = pd.read_csv(csv_path, nrows=10)
    print("✅ 成功读取前10行数据：\n")
    print(df)
except Exception as e:
    print(f"❌ 读取失败: {e}")
