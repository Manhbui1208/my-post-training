from datasets import load_dataset, Dataset, DatasetDict
import json, pathlib

DATASET = "trl-internal-testing/tldr-preference-sft-trl-style"

if __name__ == "__main__":
    raw_train = load_dataset(DATASET, split="train[:200]")
    raw_val   = load_dataset(DATASET, split="validation[:40]")

    toy_tldr = DatasetDict({"train": raw_train, "validation": raw_val})

    out_dir = pathlib.Path("data/tldr_toy_240")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in raw_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_dir / "valid.jsonl", "w", encoding="utf-8") as f:
        for ex in raw_val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    toy_tldr.save_to_disk(str(out_dir / "arrow"))

    print(" Saved dataset to:")
    print(f"  {out_dir}/train.jsonl")
    print(f"  {out_dir}/valid.jsonl")
    print(f"  {out_dir}/arrow/")