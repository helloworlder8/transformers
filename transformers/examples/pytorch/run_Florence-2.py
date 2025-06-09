import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
from torch.utils.data import Dataset

from datasets import get_dataset_config_names, load_dataset, load_from_disk

class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text
    
class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.name = "DocVQA"
        self.data = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        self.task_prompt = "<VQA>"

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = first_answer
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image

    
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# Load the model and processor
# from transformers import Florence2ForConditionalGeneration
model = AutoModelForCausalLM.from_pretrained("hub_pull/microsoft/Florence-2-base", torch_dtype=torch_dtype).to(device)
for param in model.vision_tower.parameters():
  param.is_trainable = False
processor = AutoProcessor.from_pretrained("hub_pull/microsoft/Florence-2-base")


def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers


# Create datasets
train_dataset = DocVQADataset("train")
val_dataset = DocVQADataset("validation")

# Create DataLoader
batch_size = 2
num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, #8
    collate_fn=collate_fn,
    num_workers=num_workers, #0
    shuffle=True,
    # padding=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
)


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        i = -1
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, answers = batch

            input_ids = inputs["input_ids"] #torch.Size([2, 20])
            pixel_values = inputs["pixel_values"].to(torch.float16) #torch.Size([2, 3, 768, 768])
            labels = processor.tokenizer( #->torch.Size([2, 6])
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            if i % 200 == 0:
                print(loss)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"].to(torch.float16),
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                for generated_text, answer in zip(generated_texts, answers):
                    parsed_answer = processor.post_process_generation(
                        generated_text,
                        task="<DocVQA>",
                        image_size=(
                            inputs["pixel_values"].shape[-2],
                            inputs["pixel_values"].shape[-1],
                        ),
                    )
                    print("GT:", answer)
                    print("Pred:", parsed_answer["<DocVQA>"])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                inputs, answers = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

train_model(train_loader, val_loader, model, processor, epochs=3) #数据集 模型 处理器 轮次