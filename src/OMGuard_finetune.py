import os
import json
import argparse
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from swanlab.integration.transformers import SwanLabCallback
import swanlab


tokenizer = None
processor = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with LoRA")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/Qwen3-VL-8B")
    parser.add_argument("--task_name", type=str, default="Misleading_Detection")

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="Qwen3-VL-finetune")
    parser.add_argument("--experiment_name", type=str, default="qwen3-vl-train-only")

    return parser.parse_args()


def process_func(example):
    """
    Data processing function to convert raw dataset examples into model input format.
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    text_content = input_content.split("<|vision_start|>")[0]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": text_content},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
        instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"][0])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs["pixel_values"] = torch.tensor(inputs["pixel_values"])
    inputs["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }


class MultiModalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thw = torch.stack(image_grid_thw, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def load_filtered_dataset(train_json_path, task_name):
    with open(train_json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    data = [item for item in all_data if item.get("task") == task_name]
    if len(data) == 0:
        raise ValueError(f"No samples found for task={task_name}")

    return Dataset.from_list(data), data


def build_model_processor_tokenizer(args):
    global tokenizer, processor

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_datasets(args):
    dataset, raw_data = load_filtered_dataset(args.train_json, args.task_name)

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("To save the best model, val_ratio must be in (0, 1).")

    split_dataset = dataset.train_test_split(
        test_size=args.val_ratio,
        seed=args.seed,
    )

    original_columns = split_dataset["train"].column_names

    train_dataset = split_dataset["train"].map(
        process_func,
        remove_columns=original_columns,
    )

    val_dataset = split_dataset["test"].map(
        process_func,
        remove_columns=original_columns,
    )

    return train_dataset, val_dataset, raw_data


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model_processor_tokenizer(args)
    train_dataset, val_dataset, raw_data = prepare_datasets(args)

    swanlab_callback = SwanLabCallback(
        project=args.project_name,
        experiment_name=args.experiment_name,
        config={
            "model": args.model_path,
            "task_name": args.task_name,
            "train_data_number": len(raw_data),
            "val_ratio": args.val_ratio,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultiModalCollator(tokenizer),
        callbacks=[swanlab_callback],
    )

    print("Start training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"Best model saved to: {best_model_dir}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    swanlab.finish()


if __name__ == "__main__":
    main()