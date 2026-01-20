from io import BytesIO

from datasets import Dataset, load_dataset
from PIL import Image


def load_our_dataset(file_path: str, eval_path: str) -> tuple[Dataset, Dataset]:
    ds = load_dataset(
        "eganscha/gomoku_vlm_ds",
        data_files={
            "train": file_path,
            "eval": eval_path,
        },
    )
    ds = ds.select_columns(["question", "img_bytes", "answer"])

    def preprocess_batch(batch):
        formatted_messages = []
        imgs = []
        for i in range(len(batch["question"])):
            images = []
            sample_imgs = []

            try:
                img_entries = batch["img_bytes"][i]
                if isinstance(img_entries, list):
                    for b in img_entries:
                        img = Image.open(BytesIO(b)).convert("RGB")
                        sample_imgs.append(img)
                        images.append({"type": "image", "image": img})
                else:
                    img = Image.open(BytesIO(img_entries)).convert("RGB")
                    sample_imgs.append(img)
                    images.append({"type": "image", "image": img})
            except Exception as e:
                print(f"Error decoding images at row {i}: {e}")

            formatted_messages.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a vision-language model analyzing Gomoku game positions.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": images
                        + [
                            {
                                "type": "text",
                                "text": batch["question"][i]
                                .replace(
                                    "You are a vision-language model analyzing Gomoku game positions.",
                                    "",
                                )
                                .strip(),
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": batch["answer"][i]}],
                    },
                ]
            )

            imgs.append(sample_imgs)

    dst = (
        ds["train"]
        .shuffle(seed=42)
        .map(
            preprocess_batch,
            batched=True,
            batch_size=8,
            num_proc=4,
            remove_columns=["question", "img_bytes", "answer"],
            desc="Formatting dataset with messages",
        )
    )
    dse = (
        ds["eval"]
        .shuffle(seed=42)
        .map(
            preprocess_batch,
            batched=True,
            batch_size=8,
            num_proc=4,
            remove_columns=["question", "img_bytes", "answer"],
            desc="Formatting dataset with messages",
        )
    )

    return (dst, dse)
