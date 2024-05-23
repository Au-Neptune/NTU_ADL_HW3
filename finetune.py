import argparse
import json
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from transformers.integrations import WandbCallback

from trl import SFTTrainer

from utils import get_bnb_config
from ppl import perplexity

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=1)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_freq", default=100, type=float)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    return parser.parse_args()


def create_datasets(args):
    train_data = load_dataset(
        'json',
        data_files=args.train_data,
        split='train',
    )
    train_data = train_data.train_test_split(train_size=0.3,seed=args.seed)["train"]

    print(f"Size of the train set: {len(train_data)}")

    return train_data


def formatting_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {example['instruction'][i]} ASSISTANT: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def run_training(args, train_data):
    print("Loading model and tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=get_bnb_config(),
        device_map={"": Accelerator().process_index}
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    class MyCallback(WandbCallback):
        def on_log(self, train_args, state, control, model, tokenizer, logs=None, **kwargs):
            if self._wandb is None:
                return
            if not self._initialized:
                self.setup(args, state, model)
            if state.is_world_process_zero:
                with open(args.eval_data, "r") as f:
                    data = json.load(f)

                model.eval()
                ppl = perplexity(model, tokenizer, data)
                print("Mean perplexity:", ppl["mean_perplexity"])
                self._wandb.log({"perplexity" : ppl["mean_perplexity"]})

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epoch,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optim='paged_adamw_32bit',
        gradient_checkpointing=True,
        bf16=True,
        warmup_ratio=0.2,

        logging_strategy="steps",
        logging_steps=args.log_freq,
        disable_tqdm=False,
        run_name="llama-7b-finetuned",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        formatting_func=formatting_func,
        peft_config=lora_config,
        max_seq_length=args.seq_length,
        callbacks=[MyCallback],
    )

    print("Training...")
    trainer.train()

    print("Saving")
    trainer.model.save_pretrained(args.output_dir)


def main(args):
    train_dataset = create_datasets(args)
    run_training(args, train_dataset)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)