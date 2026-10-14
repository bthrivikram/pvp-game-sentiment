from datasets import load_dataset

dataset = load_dataset("dffesalbon/dota-2-toxic-chat-data")

for split in dataset:
    df = dataset[split].to_pandas()
    df.to_csv(f"data/{split}.csv", index=False)

    print(f"{split}: {df.shape}")