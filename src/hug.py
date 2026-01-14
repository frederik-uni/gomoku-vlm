import argparse
import os
import zipfile
import tempfile
from huggingface_hub import HfApi, HfFolder, upload_file, hf_hub_download

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=folder_path)
                zipf.write(full_path, arcname)
    return zip_path

def unzip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_dir)

def upload(file_path, repo_id, token):
    if os.path.isdir(file_path):
        tmp_zip = os.path.join(tempfile.gettempdir(), os.path.basename(file_path) + ".zip")
        zip_folder(file_path, tmp_zip)
        file_path = tmp_zip

    print(f"Uploading {file_path} to {repo_id}")
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        token=token,
        repo_type="dataset"
    )
    print("complete")

def download(file_name, repo_id, token, out_dir="."):
    print(f"Downloading {file_name} from {repo_id}")
    local_path = hf_hub_download(repo_id=repo_id, filename=file_name, token=token)

    if local_path.endswith(".zip"):
        unzip_dir = os.path.join(out_dir, os.path.splitext(file_name)[0])
        os.makedirs(unzip_dir, exist_ok=True)
        unzip_file(local_path, unzip_dir)
        print(f"Unpacked to {unzip_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(local_path))
        os.replace(local_path, out_path)
        print(f"Downloaded to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="hf CLI uploader/downloader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("path", help="File/folder to upload")
    upload_parser.add_argument("repo_id", help="hf repo id")
    upload_parser.add_argument("--token", default=HfFolder.get_token(), help="token")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("file_name", help="File/zip to download")
    download_parser.add_argument("repo_id", help="hf repo id")
    download_parser.add_argument("--token", default=HfFolder.get_token(), help="token")
    download_parser.add_argument("--out_dir", default=".", help="dir to download to")

    args = parser.parse_args()

    if args.command == "upload":
        upload(args.path, args.repo_id, args.token)
    elif args.command == "download":
        download(args.file_name, args.repo_id, args.token, args.out_dir)

if __name__ == "__main__":
    main()
