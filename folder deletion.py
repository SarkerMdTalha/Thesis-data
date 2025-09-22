import os
import shutil
import stat

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree: tries to fix permission errors.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Path to your main dataset directory containing p00000 to p04051
dataset_path = 'E:/Thesis dataset/Project_CodeNet/Project_CodeNet/data'  # <-- Change if needed

# Loop through all problem folders like p00000, p00001, ..., p04051
for problem_folder in os.listdir(dataset_path):
    problem_path = os.path.join(dataset_path, problem_folder)
    
    # Only process folders (skip files or anything unexpected)
    if not os.path.isdir(problem_path):
        continue

    # Loop through language subfolders inside the problem folder
    for lang_folder in os.listdir(problem_path):
        lang_path = os.path.join(problem_path, lang_folder)

        # Delete if it's a folder and not 'python' (case-insensitive)
        if os.path.isdir(lang_path) and lang_folder.lower() != 'python':
            try:
                shutil.rmtree(lang_path, onerror=on_rm_error)
                print(f"Deleted: {lang_path}")
            except Exception as e:
                print(f"Failed to delete {lang_path}: {e}")

print("All non-python language folders removed.")
