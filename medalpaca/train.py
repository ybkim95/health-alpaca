import os
import sys
from typing import Tuple, Union

import fire
import torch
from datasets import load_dataset
from datasets import load_from_disk
from handler import DataHandler
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)



def main(
    model: str, # e.g. "decapoda-research/llama-7b-hf"
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "prompt_templates/medalpaca.json",
    model_max_length: int = 2048,  # should not exceed 2048, as LLaMA is trained with this
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    data_path: str = "medical_meadow_small.json",
    train_in_8bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    per_device_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    global_batch_size: int = 128,
    output_dir: str = "./output",
    save_total_limit: int = 2,
    eval_steps: float = 0.9,#90,
    device_map: str = "auto",
    group_by_length: bool = False,
    wandb_run_name: str = "test",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    optim: str = "adafactor", #"adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 100,
    fsdp: str = "full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap: str = "LlamaDecoderLayer",
    **kwargs
):
    """
    Trains a large language model using HuggingFace Transformers with custom configuration options.

    Args:
    model (str, optional):
        The model identifier on HuggingFace Model Hub.
    val_set_size (Union[int, float], optional):
        The proportion or number of samples to use for validation. Default is 0.1.
    prompt_template (str, optional):
        The path to the JSON file containing prompt templates. Default is "prompts/medalpaca.json".
    model_max_length (int, optional):
        The maximum length for model inputs. Default is 256.
    train_on_inputs (bool, optional):
        Whether to train on input tokens. Default is True.
    data_path (str, optional):
        The path to the dataset file. Default is "medical_meadow_small.json".
    train_in_8bit (bool, optional):
        Whether to use 8-bit training. Default is True.
    use_lora (bool, optional):
        Whether to use the Lora method. Default is True.
    lora_r (int, optional):
        The Lora method's reduction factor. Default is 8.
    lora_alpha (int, optional):
        The Lora method's alpha parameter. Default is 16.
    lora_dropout (float, optional):
        The dropout rate for Lora. Default is 0.1.
    lora_target_modules (List[str], optional):
        The target modules for Lora. Default is ["q_proj","v_proj"].
    per_device_batch_size (int, optional):
        The batch size per device. Default is 2.
    num_epochs (int, optional):
        The number of epochs for training. Default is 3.
    learning_rate (float, optional):
        The learning rate for the optimizer. Default is 2e-5.
    global_batch_size (int, optional):
        The number of samples the model needs to see until the weights get updated.
        Default is 128.
    output_dir (str, optional):
        The directory to save the model and outputs. Default is "./output".
    save_total_limit (int, optional):
        The maximum number of saved checkpoints. Default is 3.
    eval_steps (int, optional):
        The number of steps between evaluations. Default is 200.
    device_map (str, optional):
        The device placement strategy. Default is "auto".
    group_by_length (bool, optional):
        Whether to group samples by length for batch construction. Default is False.
    wandb_run_name (str, optional):
        The run name for Weights & Biases logging. Default is "test".
    use_wandb (bool, optional):
        Whether to use Weights & Biases for logging. Default is False.
    wandb_project (str, optional):
        The Weights & Biases project name. Default is "medalpaca".
    optim (str, optional):
        The optimizer to use. Default is "adamw_torch".
    lr_scheduler_type (str, optional):
        The learning rate scheduler type. Default is "cosine".
    fp16 (bool, optional):
        Whether to use mixed precision training (FP16). Default is True.
    bf16 (bool, optional):
        Whether to use mixed precision training (BF16). Default is False.
    gradient_checkpointing (bool, optional):
        Whether to use gradient checkpointing during training to reduce memory footprint
    warmup_steps (int, optional):
        The number of steps for warmup. Default is 200.
    fsdp (str, optional):
        Fully Sharded Data Parallel strategy. Only active with distributed training.
        Default is "full_shard auto_wrap"
    fsdp_transformer_layer_cls_to_wrap (optiona, str):
        The model layer to wrap for fsdp. Default is "LlamaDecoderLayer".
    **kwargs:
        additional arguments passed to the transformers.TrainingArguments"""
    model_name = model
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = global_batch_size // per_device_batch_size
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        if use_lora:
            fsdp, fsdp_transformer_layer_cls_to_wrap = "", None
    else:
        fsdp, fsdp_transformer_layer_cls_to_wrap = "", None

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    if fp16 and bf16:
        raise ValueError("At most one of fp16 and bf16 can be True, but not both.")

    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if use_lora and gradient_checkpointing:
        raise ValueError("gradient_checkpointing with LoRA training is not implemented")

    # init model
    if "llama" in model_name:
        load_model = LlamaForCausalLM
    else:
        load_model = AutoModelForCausalLM

    model = load_model.from_pretrained(
        model_name,
        load_in_8bit=train_in_8bit,
        torch_dtype=torch.float16 if any([use_lora, bf16]) else torch.float32,
        device_map=device_map,
        local_files_only=True
    )

    print("[INFO] model loaded ...")

    if train_in_8bit:
        model = prepare_model_for_int8_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    print("[INFO] tokenizer loaded ...")
    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        model_max_length=model_max_length,
        train_on_inputs=train_on_inputs,
    )
    
    print("[INFO] data_path:", data_path)
    data = load_dataset("json", data_files=data_path)

    
    print("[INFO] data loaded ...")

    val_set_size = 0.1

    # def tokenize(batch):
    #     tokens = tokenizer(batch['text'], padding=True, truncation=True, max_length=256)
    #     # tokens['labels'] = labels.str2int(batch['labels'])
    #     return tokens


    if val_set_size > 0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            .map(data_handler.generate_and_tokenize_prompt)
            # .map(tokenize, batched=True)
        )
    else:
        data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_steps=eval_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        fsdp=fsdp,
        fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
        **kwargs
    )

    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0 else None,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    
    # model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)
