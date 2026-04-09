import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json


def process_func(example):
    """
    将数据集进行预处理
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
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
   
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
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# 使用本地Qwen3-VL模型
model_path = "/public/lifanxiao/workspace/local_models/Qwen3-VL-8B-Instruct"

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)

model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件，只使用Misleading_Detection任务的数据
train_json_path = "/public/lifanxiao/workspace/Omission_misleading/datas/train_subset_finetune.json"
test_json_path = "/public/lifanxiao/workspace/Omission_misleading/datas/test_subset_finetune.json"

# 读取训练数据并筛选Misleading_Detection任务
with open(train_json_path, 'r') as f:
    all_train_data = json.load(f)
    train_data = [item for item in all_train_data if item.get("task") == "Misleading_Detection"]

# 读取测试数据并筛选Misleading_Detection任务
with open(test_json_path, 'r') as f:
    all_test_data = json.load(f)
    test_data = [item for item in all_test_data if item.get("task") == "Misleading_Detection"]

# 保存筛选后的数据
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)

with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json("data_vl_train.json")
train_dataset = train_ds.map(process_func)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen3-VL-8B",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
        
# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-VL-finetune",
    experiment_name="qwen3-vl-misleading-detection",
    config={
        "model": model_path,
        "dataset": "Misleading_Detection",
        "train_data_number": len(train_data),
        "test_data_number": len(test_data),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

# 检查是否已有训练好的checkpoint，如果有则跳过训练
checkpoint_dir = "./output/Qwen3-VL-8B"
preferred_checkpoint = os.path.join(checkpoint_dir, "checkpoint-1300")
skip_training = os.path.exists(preferred_checkpoint)

if skip_training:
    print(f"检测到已有训练好的checkpoint: checkpoint-1300")
    print("跳过训练，直接进入测试模式...")
else:
    # 配置Trainer
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    # 开启模型训练
    print("开始训练模型...")
    trainer.train()

# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 推理模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型（优先使用损失最低的checkpoint-1300）
# checkpoint_dir 和 preferred_checkpoint 已在上面定义，这里直接使用
checkpoint_path = None

# 优先使用checkpoint-1300（损失最低）
if os.path.exists(preferred_checkpoint):
    checkpoint_path = os.path.abspath(preferred_checkpoint)
    print(f"使用最佳checkpoint（损失最低）: checkpoint-1300")
else:
    # 如果checkpoint-1300不存在，查找最新的checkpoint
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # 按checkpoint编号排序，取最新的
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            checkpoint_path = os.path.abspath(checkpoint_path)  # 转换为绝对路径
            print(f"checkpoint-1300不存在，使用最新checkpoint: {os.path.basename(checkpoint_path)}")

if checkpoint_path and os.path.exists(checkpoint_path):
    val_peft_model = PeftModel.from_pretrained(model, model_id=checkpoint_path, config=val_config)
else:
    print(f"Warning: Checkpoint not found at {checkpoint_dir}, using trained model for testing.")
    val_peft_model = peft_model  # 使用训练后的模型

# 读取测试数据
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    # 提取文本部分
    text_content = input_image_prompt.split("<|vision_start|>")[0]
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": origin_image_path
            },
            {
            "type": "text",
            "text": text_content
            }
        ]}]
    
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])

    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

swanlab.log({"Prediction": test_image_list})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()

