import os

label_dirs = [
    './dataset/train/labels',
    './dataset/valid/labels',
    './dataset/test/labels'
]

def fix_labels():
    fixed_files = []

    for folder in label_dirs:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                changed = False
                new_lines = []
                for line in lines:
                    if line.strip() == '':
                        new_lines.append(line)
                        continue

                    parts = line.strip().split()
                    # Replace the class ID (first element) with '0'
                    if parts[0] != '0':
                        parts[0] = '0'
                        changed = True
                    new_line = ' '.join(parts) + '\n'
                    new_lines.append(new_line)

                if changed:
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    fixed_files.append(file_path)

    if fixed_files:
        print("Fixed class IDs in these files:")
        for file in fixed_files:
            print(f" - {file}")
    else:
        print("No files needed fixing.")

if __name__ == "__main__":
    fix_labels()
