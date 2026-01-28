import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoProcessor, EarlyStoppingCallback
from trl.trainer.sft_trainer import SFTTrainer

from .args import parse_args
from .configs import init_lora, init_train, modules, target
from .dataset import load_our_dataset
from .setup import init_model, init_save
from .util import get_sorted_adapter_paths

if __name__ == "__main__":
    args = parse_args()

    adapter_paths = get_sorted_adapter_paths(args.output_dir)
    model = init_model(args.model_id)

    if adapter_paths:
        print(f"Found {len(adapter_paths)} adapters to apply sequentially.")
        for adapter_path in adapter_paths:
            print(f"  -> Applying adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
    else:
        print("No previous adapters found. Starting from base model.")

    if args.peft:
        # model = PeftModel.from_pretrained(model, args.peft).eval()
        model.load_adapter(args.peft, adapter_name="vision_adapter", is_trainable=False)
        model.set_adapter("vision_adapter")

    new_output_dir = init_save(args.output_dir)
    print(f"New adapter will be saved to: {new_output_dir}")

    final_dir = os.path.join(new_output_dir, "final-adapter")

    dst, dse = load_our_dataset(args.data_file, args.eval_path)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dst,
        eval_dataset=dse,
        args=init_train(
            new_output_dir,
            args.num_epochs,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.learning_rate,
        ),
        peft_config=init_lora(args.lora_r, target(args.mode), modules(args.mode)),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        processing_class=processor,
    )

    trainer.train()

    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    print("Saved to:", final_dir)
