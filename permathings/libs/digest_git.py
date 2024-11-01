import argparse
import os
import tempfile
import zipfile
from urllib.parse import urlparse
import requests
import logging
from math import floor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def download_repo(url):
    if url.endswith('.git'):
        url = url[:-4]
    if not url.endswith('/archive/main.zip'):
        url += '/archive/main.zip'  # Try 'main' branch
    
    logger.info(f"Attempting to download from URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        # If 'main' branch doesn't exist, try 'master'
        url = url.replace('/main.zip', '/master.zip')
        logger.info(f"Main branch not found. Trying master: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download repository: HTTP status code {response.status_code}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        temp_file.write(response.content)
        logger.info(f"Repository downloaded to temporary file: {temp_file.name}")
        return temp_file.name

def extract_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Extracting ZIP file to temporary directory: {temp_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir
    except zipfile.BadZipFile:
        logger.error(f"Failed to extract ZIP file: {zip_path}")
        raise

def get_tree_structure(path, max_folder_depth, depth_augmented_limit_per_folder):
    tree = []
    for root, dirs, files in os.walk(path):
        # Include hidden directories
        dirs[:] = [d for d in dirs]
        level = root.replace(path, '').count(os.sep)
        if level > max_folder_depth:
            if level == max_folder_depth + 1:
                indent = ' ' * 4 * max_folder_depth
                tree.append(f"{indent}... (deeper folders omitted)")
            continue
        
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        
        file_limit = max(2, floor(depth_augmented_limit_per_folder / (2 ** level)))
        
        subindent = ' ' * 4 * (level + 1)
        # Include hidden files
        all_files = files + [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.startswith('.')]
        if len(all_files) > file_limit:
            files_to_show = all_files[:file_limit // 2] + all_files[-file_limit // 2:]
            for file in files_to_show:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                tree.append(f"{subindent}{file} ({human_readable_size(size)})")
            
            omitted_files = len(all_files) - len(files_to_show)
            omitted_size = sum(os.path.getsize(os.path.join(root, f)) for f in all_files if f not in files_to_show)
            tree.append(f"{subindent}... ({omitted_files} files omitted, total size: {human_readable_size(omitted_size)})")
        else:
            for file in all_files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                tree.append(f"{subindent}{file} ({human_readable_size(size)})")
    
    return '\n'.join(tree)

def process_file(file_path, max_size, demarcation_string, ignored_extensions, whitelisted_extensions, base_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ignored_extensions and ext in ignored_extensions:
        return None
    if whitelisted_extensions and ext not in whitelisted_extensions:
        return None

    relative_file_path = os.path.relpath(file_path, base_path)
    file_size = os.path.getsize(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) > max_size:
            half_size = max_size // 2
            truncated_chars = len(content) - max_size
            content = f"{content[:half_size]}...[truncated middle {truncated_chars} characters]...{content[-half_size:]}"

        return f"{demarcation_string} {relative_file_path} ({human_readable_size(file_size)}):\n{content}\n"
    except UnicodeDecodeError:
        return f"{demarcation_string} {relative_file_path} ({human_readable_size(file_size)}):\n(Binary file)\n"

def process_repo(repo_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, depth_augmented_limit_per_folder, max_folder_depth, lower_bound_limit_per_folder=2):
    result = []
    skipped_folders_count = 0
    total_skipped_size = 0
    logger.info(f"Processing repository at path: {repo_path}")
    base_path = repo_path

    for root, dirs, files in os.walk(repo_path):
        # Include hidden directories
        dirs[:] = [d for d in dirs]
        level = root.replace(repo_path, '').count(os.sep)
        if level > max_folder_depth:
            skipped_folders_count += 1
            continue

        file_limit = max(lower_bound_limit_per_folder, floor(depth_augmented_limit_per_folder / (2 ** level)))
        files_processed = 0
        # Include hidden files
        all_files = files + [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.startswith('.')]
        files_to_process = min(max(file_limit, lower_bound_limit_per_folder), len(all_files))

        folder_content = []
        folder_skipped_size = 0
        for file in all_files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            if files_processed >= files_to_process:
                folder_skipped_size += file_size
                continue
            
            processed = process_file(file_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, base_path)
            if processed:
                folder_content.append(processed)
                files_processed += 1
            else:
                folder_skipped_size += file_size
        
        total_skipped_size += folder_skipped_size
        
        if files_processed < len(all_files):
            skipped = len(all_files) - files_processed
            relative_path = os.path.relpath(root, repo_path)
            skipped_info = f"{demarcation_string} {relative_path}: {skipped} file(s) skipped in this folder (total size: {human_readable_size(folder_skipped_size)})\n"
            folder_content.append(skipped_info)
        elif files_processed == 0:
            relative_path = os.path.relpath(root, repo_path)
            empty_folder_info = f"{demarcation_string} {relative_path}: Empty folder or all files filtered\n"
            folder_content.append(empty_folder_info)
        
        result.extend(folder_content)

    if skipped_folders_count > 0:
        skipped_folders_info = f"{demarcation_string} {skipped_folders_count} folder(s) skipped due to depth limit\n"
        result.append(skipped_folders_info)

    total_skipped_info = f"{demarcation_string} Total size of skipped files: {human_readable_size(total_skipped_size)}\n"
    result.append(total_skipped_info)

    return ''.join(result)

def main(url_or_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, depth_augmented_limit_per_folder, max_folder_depth, output_path):
    parsed_url = urlparse(url_or_path)
    
    if parsed_url.scheme in ('http', 'https'):
        if url_or_path.endswith('.zip'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                response = requests.get(url_or_path)
                temp_file.write(response.content)
                zip_path = temp_file.name
        else:
            zip_path = download_repo(url_or_path)
    elif os.path.isfile(url_or_path) and url_or_path.endswith('.zip'):
        zip_path = url_or_path
    else:
        raise ValueError("Invalid input. Please provide a GitHub repo URL, a zip file URL, or a local zip file path.")

    repo_name = os.path.basename(url_or_path.rstrip('/'))
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    temp_dir = extract_zip(zip_path)
    
    # Find the actual repository directory
    repo_path = temp_dir
    try:
        subdirs = next(os.walk(temp_dir))[1]
        if subdirs:
            repo_path = os.path.join(temp_dir, subdirs[0])
    except StopIteration:
        logger.warning("The repository appears to be empty.")
    
    tree_structure = get_tree_structure(repo_path, max_folder_depth, depth_augmented_limit_per_folder)
    processed_content = process_repo(repo_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, depth_augmented_limit_per_folder, max_folder_depth)
    
    result = f"Repository: {repo_name}\n\nFile Structure:\n{tree_structure}\n\nFile Contents:\n{processed_content}"
    
    if output_path:
        if os.path.isdir(output_path):
            # If output_path is a directory, use the original file naming scheme
            output_file = os.path.join(output_path, f"{repo_name}_processed.txt")
        else:
            # If output_path is not a directory, create the necessary directories
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_file = output_path
    else:
        # If no output_path is provided, use the default naming in the current directory
        output_file = f"{repo_name}_processed.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    logger.info(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GitHub repository or zip file.")
    parser.add_argument("url_or_path", help="GitHub repo URL, zip file URL, or local zip file path")
    parser.add_argument("--max_size_per_file", type=int, default=2000, help="Maximum number of characters to include per file")
    parser.add_argument("--demarcation_string", default="\n-----new_file-----\n", help="String to use between files")
    parser.add_argument("--ignored_extensions", nargs="+", help="File extensions to ignore")
    parser.add_argument("--whitelisted_extensions", nargs="+", help="File extensions to include (if specified, only these will be processed)")
    parser.add_argument("--depth_augmented_limit_per_folder", type=int, default=40, help="Starting limit for files per folder, halved with each level of depth")
    parser.add_argument("--max_folder_depth", type=int, default=8, help="Maximum folder depth to process")
    parser.add_argument("--lower_bound_limit_per_folder", type=int, default=2, help="Minimum number of files to show per folder")
    parser.add_argument("--output_path", help="Custom output file path or directory")
    
    args = parser.parse_args()
    
    main(args.url_or_path, args.max_size_per_file, args.demarcation_string, args.ignored_extensions, args.whitelisted_extensions, args.depth_augmented_limit_per_folder, args.max_folder_depth, args.output_path)
