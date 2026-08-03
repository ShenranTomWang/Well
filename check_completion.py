import os
from pathlib import Path


def count_lines(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def remove_empty_files(directory: Path) -> list[Path]:
    removed_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = Path(root) / file_name
            try:
                if file_path.stat().st_size == 0:
                    file_path.unlink()
                    removed_files.append(file_path)
            except FileNotFoundError:
                continue
    return removed_files


def remove_empty_dirs(directory: Path) -> list[Path]:
    removed_dirs = []
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                dir_path.rmdir()
                removed_dirs.append(dir_path)
            except OSError:
                continue
    return removed_dirs


def main() -> None:
    directory_input = input("Enter directory path: ").strip()
    expected_lines_input = input("Enter expected number of lines: ").strip()

    directory = Path(directory_input).expanduser()
    if not directory.is_dir():
        print(f"Not a valid directory: {directory}")
        return

    try:
        expected_lines = int(expected_lines_input)
    except ValueError:
        print(f"Invalid number: {expected_lines_input}")
        return

    removed_files = remove_empty_files(directory)
    removed_dirs = remove_empty_dirs(directory)
    if removed_files:
        print(f"Removed {len(removed_files)} empty files.")
        for file_path in removed_files:
            print(f"  {file_path}")
    if removed_dirs:
        print(f"Removed {len(removed_dirs)} empty directories.")
        for dir_path in removed_dirs:
            print(f"  {dir_path}")

    jsonl_files = sorted(
        Path(root) / file_name
        for root, _, files in os.walk(directory)
        for file_name in files
        if file_name.endswith(".jsonl")
    )
    if not jsonl_files:
        print("No .jsonl files found.")
        return

    mismatches = []
    for file_path in jsonl_files:
        line_count = count_lines(file_path)
        if line_count != expected_lines:
            mismatches.append((file_path, line_count))

    if not mismatches:
        print(f"All .jsonl files have exactly {expected_lines} lines.")
        return

    print("Files that do not match:")
    for file_name, line_count in mismatches:
        print(f"{file_name}: {line_count}")


if __name__ == "__main__":
    main()
