import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, PeftModel, TaskType
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)


def extract_json_string(s: str) -> str:
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace == -1 or last_brace == -1:
        raise ValueError("No valid JSON found in the string")
    return s[first_brace:last_brace + 1]


def read_prompt_from_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def load_finetuned_model(checkpoint_path, base_model_path, device="cuda:0"):
    print(f"Loading model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(base_model_path)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights: {checkpoint_path}")
    val_config = LoraConfig(
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
        inference_mode=True,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    model = PeftModel.from_pretrained(
        base_model,
        model_id=checkpoint_path,
        config=val_config,
    )
    model.eval()

    return model, processor


def generate_qwen_3_vl_8b_response(
    args,
    model,
    processor,
    prompt: str,
    image_path: str,
    system: str = "",
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def generate_qwen_3_language_8b_response(
    args,
    model,
    processor,
    prompt: str,
    system: str = "",
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main(args):
    
    # ❕ Placeholder for the actual data loading and processing logic
    Misleading_Caption = ""
    Full_Article_Context = ""
    Detection_Rationales = ""
    image_path = f"/datas/test_img/xxx.jpg"


   
    if args.correction_type == "free_form":
        correction_tmp = read_prompt_from_md("../prompts/free_form.md")
    elif args.correction_type == "mini_edit":
        correction_tmp = read_prompt_from_md("../prompts/mini_edit.md")

    prompt = correction_tmp.replace("{{NEWS_HEADLINE}}", Misleading_Caption).replace("{{NEWS_CONTEXT}}", Full_Article_Context).replace("{{MISLEADING_REASON}}", Detection_Rationales).replace("{{limit_words}}", str(args.limit_words))

    # ❕ Placeholder for the actual model paths
    base_model_path = "local_path/Qwen3-VL-8B-Instruct"
    checkpoint_path = "output/checkpoint-qwen-ft"

    local_models, local_processors = load_finetuned_model(
        checkpoint_path,
        base_model_path,
        device=args.device,
    )
    print("Finish Loading Model.\n")

    correction_response = generate_qwen_3_vl_8b_response(
                    args,
                    local_models,
                    local_processors,
                    prompt,
                    image_path,
                    system="",
                )
    correction_response = extract_json_string(correction_response)

    Misleading_Cause = json.loads(correction_response).get("misleading_cause", "")
    Suggested_Improvement = json.loads(correction_response).get("suggested_improvement", "")
    Rewriten_Caption = json.loads(correction_response).get("rewritten_caption", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the last time",
    )
    parser.add_argument("--content", type=str, default="caption")
    parser.add_argument("--model", type=str, default="OMGuard")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--limit_words", type=int, default=3)
    parser.add_argument("--correction_type", type=str, default="free_form")
    args = parser.parse_args()
    main(args)