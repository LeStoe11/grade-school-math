import torch as th
from datasets import load_dataset

from dataset import get_examples, GSMDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def main():
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    train_examples = load_dataset("gsm8k", "main")["train"]
    train_dset = GSMDataset(tokenizer, train_examples)

    device = th.device("cuda")
    config = T5Config.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small", config=config)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    model.save_pretrained("model_ckpts/")


if __name__ == "__main__":
    main()
