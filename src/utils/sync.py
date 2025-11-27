import os

from huggingface_hub import Repository


def load_save(local_dir: str = "./models", repo_name: str = "username/repo"):
    if not os.path.exists(local_dir):
        repo = Repository(local_dir=local_dir, clone_from=repo_name)
    else:
        repo = Repository(local_dir=local_dir)
    repo.git_pull(rebase=True, lfs=True)
    repo.git_add(pattern="*", auto_lfs_track=True)

    repo.git_commit("update model files")
    repo.git_push()
