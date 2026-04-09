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
    datas = json.load(open(f"datas/test_data.json"))
    all_results = {}
    sample_count = 0
    resume_count = 0

    if args.resume:
        pre_result_path = open(
            f"./results/finetune_misleading_eval_{args.data_type}_{args.model}.json",
            encoding="utf-8",
        )
        pre_results = json.load(pre_result_path)
        for key in pre_results:
            all_results[key] = pre_results[key]
            resume_count += 1

    CI_ih_View_Tmp = read_prompt_from_md("../prompts/reader_view_ih.md")
    CI_context_View_Tmp = read_prompt_from_md("../prompts/reader_view_context.md")
    misleading_tmp = read_prompt_from_md("../prompts/compare_misleading.md")

    base_model_path = "local_path/Qwen3-VL-8B-Instruct"
    checkpoint_path = "output/checkpoint-qwen-ft"

    local_models, local_processors = load_finetuned_model(
        checkpoint_path,
        base_model_path,
        device=args.device,
    )
    print("Finish Loading Model.\n")

    for key in datas:
        sample_count += 1
        if sample_count <= resume_count:
            continue

        image_path = f"/datas/test_img/{key}.jpg"

        caption = datas[key]["caption"]
        article_context = datas[key]["Article_Context"]

        print("*" * 20, f"Sample {sample_count} / {len(datas)}", "*" * 20)
        print("===================================")
        print("Vanilla caption: ", caption)
        print("article_context: ", article_context)

        for repeat in range(5):
            try:
                ih_reader_view_response = generate_qwen_3_vl_8b_response(
                    args,
                    local_models,
                    local_processors,
                    CI_ih_View_Tmp.replace("{{NEWS_HEADLINE}}", caption),
                    image_path,
                    system="",
                )
                ih_reader_view_response = extract_json_string(ih_reader_view_response)

                context_reader_view_response = generate_qwen_3_language_8b_response(
                    args,
                    local_models,
                    local_processors,
                    CI_context_View_Tmp.replace("{{CONTEXT}}", article_context),
                    system="",
                )
                context_reader_view_response = extract_json_string(
                    context_reader_view_response
                )

                all_view = ih_reader_view_response + "\n" + context_reader_view_response

                misleading_response = generate_qwen_3_vl_8b_response(
                    args,
                    local_models,
                    local_processors,
                    misleading_tmp.replace("{{NEWS_HEADLINE}}", caption)
                    .replace("{{CONTEXT}}", article_context)
                    .replace("{{READER_INFER}}", all_view),
                    image_path,
                    system="",
                )
                misleading_response = extract_json_string(misleading_response)

                print("Reader_views: ", all_view)
                print("===================================")
                print("Misleading: ", misleading_response)
                print("===================================")

                break
            except:
                continue

        if repeat == 4:
            print("max try... Done")

        all_results[key] = {
            "caption": caption,
            "article_context": article_context,
            "image_path": image_path,
            "Reader_View": all_view,
            "Misleading": misleading_response,
        }

        if sample_count % 5 == 0 and sample_count > 0:
            with open(
                f"./results/finetune_misleading_eval_data_{args.data_type}_subset_{args.model}.json",
                "w",
                encoding="utf-8",
            ) as output_file:
                json.dump(all_results, output_file, ensure_ascii=False, indent=4)

    with open(
        f"./results/finetune_misleading_eval_data_{args.data_type}_subset_{args.model}.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(all_results, output_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the last time",
    )
    parser.add_argument("--content", type=str, default="caption")
    parser.add_argument("--model", type=str, default="finetuned_qwen3-vl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_type", type=str, default="test")
    args = parser.parse_args()
    main(args)