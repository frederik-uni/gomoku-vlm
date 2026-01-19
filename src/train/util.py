from pathlib import Path


def get_sorted_adapter_paths(root: str) -> list[str]:
    p = Path(root)
    if not p.exists():
        return []

    lora_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("lora_")]
    if not lora_dirs:
        return []

    sorted_lora_dirs = sorted(lora_dirs, key=lambda d: int(d.name.split("_")[-1]))

    adapter_paths = []
    for lora_dir in sorted_lora_dirs:
        adapter_path = lora_dir / "final-adapter"
        if adapter_path.exists() and adapter_path.is_dir():
            adapter_paths.append(str(adapter_path))
    return adapter_paths
