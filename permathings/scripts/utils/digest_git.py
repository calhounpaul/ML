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
        level = root.replace(path, '').count(os.sep)
        if level > max_folder_depth:
            if level == max_folder_depth + 1:
                indent = ' ' * 4 * max_folder_depth
                tree.append(f"{indent}... (deeper folders omitted)")
            continue
        
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        
        file_limit = max(2, floor(depth_augmented_limit_per_folder / (2 ** level)))
        
        if file_limit > 5:
            files_to_show = files[:file_limit]
            subindent = ' ' * 4 * (level + 1)
            for file in files_to_show:
                tree.append(f"{subindent}{file}")
            
            if len(files) > file_limit:
                remaining_files = len(files) - file_limit
                tree.append(f"{subindent}... ({remaining_files} more file(s))")
        else:
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                tree.append(f"{subindent}{file}")
    
    return '\n'.join(tree)

def process_file(file_path, max_size, demarcation_string, ignored_extensions, whitelisted_extensions, base_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ignored_extensions and ext in ignored_extensions:
        return None
    if whitelisted_extensions and ext not in whitelisted_extensions:
        return None

    # Strip the base path from the file path and calculate the relative file path
    relative_file_path = file_path.replace(base_path, "")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) > max_size:
            half_size = max_size // 2
            truncated_chars = len(content) - max_size
            content = f"{content[:half_size]}...[truncated middle {truncated_chars} characters]...{content[-half_size:]}"

        return f"{demarcation_string} {relative_file_path}:\n{content}\n"
    except UnicodeDecodeError:
        # Handle binary files or non-text files
        return f"{demarcation_string} {relative_file_path}:\n(Binary file)\n"

def process_repo(repo_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, depth_augmented_limit_per_folder, max_folder_depth, lower_bound_limit_per_folder=2):
    result = []
    logger.info(f"Processing repository at path: {repo_path}")
    base_path = repo_path  # Save the base path to strip from file paths
    for root, dirs, files in os.walk(repo_path):
        level = root.replace(repo_path, '').count(os.sep)
        if level > max_folder_depth:
            continue

        file_limit = max(lower_bound_limit_per_folder, floor(depth_augmented_limit_per_folder / (2 ** level)))
        files_processed = 0
        for file in files:
            if files_processed >= file_limit:
                break
            file_path = os.path.join(root, file)
            processed = process_file(file_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, base_path)
            if processed:
                result.append((file_path, processed))
                files_processed += 1
        
        if files_processed < len(files):
            skipped = len(files) - files_processed
            result.append((root, f"{demarcation_string} ... {skipped} more file(s) in this folder were skipped due to folder limit.\n"))

    # Sort by depth (files closer to root come first)
    result.sort(key=lambda x: x[0].count(os.sep))
    return ''.join(content for _, content in result)

def main(url_or_path, max_size_per_file, demarcation_string, ignored_extensions, whitelisted_extensions, depth_augmented_limit_per_folder, max_folder_depth):
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
    
    args = parser.parse_args()
    
    main(args.url_or_path, args.max_size_per_file, args.demarcation_string, args.ignored_extensions, args.whitelisted_extensions, args.depth_augmented_limit_per_folder, args.max_folder_depth)
