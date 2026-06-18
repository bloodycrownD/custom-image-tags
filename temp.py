import os,torch
from pathlib import Path

def count_files_in_directory():
    root_dir = Path('/media/bloodycrown/我的硬盘/Pictures/Storage/classification')
    total_files = 0
    author_stats = {}

    # 遍历每个作者目录
    for author_dir in root_dir.iterdir():
        if not author_dir.is_dir():
            continue
        author_id = author_dir.name
        author_stats[author_id] = {'good': 0, 'keep': 0, 'trash': 0}

        # 处理每个分类目录
        for category in ['good', 'keep', 'trash']:
            category_dir = author_dir / category
            if not category_dir.is_dir():
                continue  # 跳过不存在的分类目录

            # 递归统计所有文件
            file_count = sum(1 for file_path in category_dir.rglob('*') if file_path.is_file())
            author_stats[author_id][category] = file_count
            total_files += file_count

    # 输出结果
    print(f"Total files: {total_files}")
    print(author_stats)
    # for author_id, counts in author_stats.items():
    #     print(f"Author {author_id}:")
    #     for category, count in counts.items():
    #         print(f"  {category}: {count}")

if __name__ == "__main__":
    count_files_in_directory()
