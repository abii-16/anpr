import os

# Paths to your label folders
label_dirs = [
    './dataset/train/labels',
    './dataset/valid/labels',
    './dataset/test/labels'
]

def check_labels():
    invalid_files = []

    for folder in label_dirs:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    if line.strip() == '':
                        continue
                    parts = line.strip().split()
                    class_id = parts[0]
                    if class_id != '0':
                        invalid_files.append((file_path, line_num, class_id))
                        break  # no need to check further lines in this file

    if invalid_files:
        print("Files with invalid class IDs found:")
        for file_path, line_num, class_id in invalid_files:
            print(f" - {file_path} (line {line_num}): class ID = {class_id}")
    else:
        print("All label files have class ID = 0 only. No issues found.")

if __name__ == "__main__":
    check_labels()
