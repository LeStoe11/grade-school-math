import csv

import torch as th
from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import T5Tokenizer, T5ForConditionalGeneration


def main():
    device = th.device("cuda")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("model_ckpts")
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")

    results = []
    for i in range(100):
        qn = test_examples[i]["question"]
        sample_len = 100
        print(qn.strip())
        input_ids = tokenizer(qn, return_tensors="pt").input_ids.to(device)
        output = model.generate(input_ids, max_new_tokens=sample_len)[0]
        output = tokenizer.decode(output, skip_special_tokens=True)
        print(output)
        results.append([qn, output])

    with open("samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)


if __name__ == "__main__":
    main()
