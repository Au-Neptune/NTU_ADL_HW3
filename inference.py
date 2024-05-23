import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--peft_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--output_path", type=str, default="./prediction.json")
    return parser.parse_args()


def predict(model, tokenizer, data):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    ids = [x["id"] for x in data]

    data_outputs = []
    for i in tqdm(range(data_size)):
        model_inputs = tokenizer(instructions[i], add_special_tokens=False, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs)
        
        model_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        split_outputs = model_outputs[0].split("ASSISTANT:")[1].strip()
        data_outputs.append({"id": ids[i], "output": split_outputs})
    return data_outputs

if __name__ == "__main__":
    args = get_args()

    # Load model
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.data, "r") as f:
        data = json.load(f)

    model.eval()
    outputs = predict(model, tokenizer, data)
    

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
