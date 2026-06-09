import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# This script illustrates the exact code used to fine-tune a pre-trained LLM
# (e.g. Llama-3 or Mistral) on an H200 GPU using LoRA (PEFT) for medical chats.

def main():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Base model
    dataset_name = "Kent0n-Li/ChatDoctor"            # Medical conversation dataset
    output_dir = "./lora_output"
    
    print(f"Initializing LoRA Fine-Tuning Pipeline for base model: {model_id}")
    
    # ----------------------------------------------------
    # GPU OPTIMIZATION: 4-bit Quantization (Day 9 Concept)
    # ----------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Speeds up computation on H200
    )
    
    # Load base model & tokenizer
    print("Loading base model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" # Auto-distribute across available GPUs
    )
    
    # Prepare model for PEFT training (freezes base, sets up checkpoints)
    model = prepare_model_for_kbit_training(model)
    
    # ----------------------------------------------------
    # ALGORITHM 3: PEFT / LoRA Adapter Setup (Day 7 Concept)
    # ----------------------------------------------------
    peft_config = LoraConfig(
        r=8,                       # Rank: lower = fewer parameters to train
        lora_alpha=16,             # Scaling factor for adapters
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"      # Causal Language Modeling task
    )
    
    # Wrap base model with PEFT adapters
    model = get_peft_model(model, peft_config)
    print("PEFT/LoRA adapter wrapped successfully!")
    model.print_trainable_parameters() # Prints fraction of trainable weights (~0.1%)
    
    # Load dataset
    print(f"Loading medical dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split="train[:5000]") # Subset for demo
    except Exception as e:
        print(f"Dataset download failed ({e}). Generating mock medical dialogue dataset...")
        # Fallback to local mock data if internet is slow or dataset is private
        mock_data = {
            "instruction": ["I have a fever and cough.", "My knee is swelling after running."],
            "input": ["", ""],
            "output": [
                "You may be experiencing a viral respiratory infection like Influenza or COVID-19. Rest and monitor your temperature.",
                "This could be joint inflammation or a minor meniscus strain. Apply ice, keep it elevated, and rest."
            ]
        }
        from datasets import Dataset as DatasetHF
        dataset = DatasetHF.from_dict(mock_data)

    def preprocess_function(examples):
        # Format the text prompt for instruction fine-tuning
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"### System: You are a helpful medical assistant.\n### User: {inst}\n{inp}\n### Assistant: {out}"
            texts.append(text)
        
        # Tokenize inputs
        tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
        
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    # ----------------------------------------------------
    # TRAINING CONFIGURATION: Day 9 GPU Optimizations
    # ----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        # GPU Optimizations
        bf16=True,                          # Use BFloat16 mixed precision (saves memory)
        fp16=False,
        optim="paged_adamw_8bit",           # Memory paged optimizer (saves memory)
        gradient_checkpointing=True,        # Backprop memory saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args
    )
    
    print("Fine-tuning pipeline prepared. Ready to execute trainer.train() on GPU.")
    print("To run the training, uncomment the line below:")
    # trainer.train()
    
if __name__ == "__main__":
    main()
