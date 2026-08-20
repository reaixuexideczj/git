"""按文件类型整理指定文件夹。

默认只预览将要执行的操作；传入 --execute 后才会真正移动文件。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CATEGORY_BY_EXTENSION = {
    # 图片
    ".jpg": "images",
    ".jpeg": "images",
    ".png": "images",
    ".gif": "images",
    ".bmp": "images",
    ".webp": "images",
    # 文档
    ".txt": "documents",
    ".md": "documents",
    ".pdf": "documents",
    ".doc": "documents",
    ".docx": "documents",
    ".ppt": "documents",
    ".pptx": "documents",
    ".xls": "documents",
    ".xlsx": "documents",
    # 代码
    ".py": "code",
    ".ipynb": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".cpp": "code",
    ".c": "code",
    ".html": "code",
    ".css": "code",
    # 压缩包
    ".zip": "archives",
    ".rar": "archives",
    ".7z": "archives",
    ".tar": "archives",
    ".gz": "archives",
    # 音视频
    ".mp3": "media",
    ".wav": "media",
    ".mp4": "media",
    ".mov": "media",
    ".avi": "media",
}


def category_for(path: Path) -> str:
    """根据扩展名返回分类目录名。"""
    return CATEGORY_BY_EXTENSION.get(path.suffix.lower(), "others")


def unique_destination(destination: Path) -> Path:
    """目标文件已存在时自动添加序号，避免覆盖。"""
    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def organize(directory: Path, execute: bool = False) -> tuple[int, int]:
    """整理目录中的第一层文件，返回（处理数，跳过数）。"""
    directory = directory.expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"目标不是文件夹：{directory}")

    script_path = Path(__file__).resolve()
    processed = 0
    skipped = 0

    print(f"目标目录：{directory}")
    print("执行模式：", "正式移动" if execute else "仅预览（不会移动）")
    print("-" * 60)

    # 先生成静态列表，避免遍历时受新建目录影响。
    for source in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not source.is_file() or source.resolve() == script_path:
            skipped += 1
            continue

        category = category_for(source)
        category_directory = directory / category
        destination = unique_destination(category_directory / source.name)

        action = "移动" if execute else "预览"
        print(f"[{action}] {source.name} -> {category}/{destination.name}")

        if execute:
            category_directory.mkdir(exist_ok=True)
            shutil.move(str(source), str(destination))

        processed += 1

    print("-" * 60)
    print(f"处理文件：{processed} 个；跳过目录或脚本自身：{skipped} 个")

    if not execute:
        print("提示：确认预览无误后，加 --execute 才会真正移动文件。")

    return processed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按扩展名将文件分类到 images、documents、code 等目录。"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="需要整理的文件夹，默认为当前目录。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行移动；不添加时仅预览。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organize(Path(args.directory), execute=args.execute)


if __name__ == "__main__":
    main()
