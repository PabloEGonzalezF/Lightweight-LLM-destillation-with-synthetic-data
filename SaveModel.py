import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STUDENT_MODEL_ID = "unsloth/granite-4.0-350m"
STUDENT_ADAPTER_DIR = "./student_lora_adapter"

OUTPUT_LORA_ONLY     = "./saved_lora_adapter"
OUTPUT_MERGED        = "./saved_model_merged"
OUTPUT_BASE          = "./saved_model_base"

SAVE_LORA_ONLY = True    # saves only the LoRA adapter weights
SAVE_MERGED    = True    # saves base model and the LoRA merged into a single model
SAVE_BASE      = True    # saves the original base model without any adapter


def save_lora_only():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base_model, STUDENT_ADAPTER_DIR)
    model = model.to(torch.bfloat16)

    model.save_pretrained(OUTPUT_LORA_ONLY)
    tokenizer.save_pretrained(OUTPUT_LORA_ONLY)
    print(f"Saved in {OUTPUT_LORA_ONLY}\n")


def save_merged():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base_model, STUDENT_ADAPTER_DIR)
    model = model.to(torch.bfloat16)
    model = model.merge_and_unload()
    model.eval()

    model.save_pretrained(OUTPUT_MERGED)
    tokenizer.save_pretrained(OUTPUT_MERGED)
    print(f"Saved in {OUTPUT_MERGED}\n")


def save_base():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    model.save_pretrained(OUTPUT_BASE)
    tokenizer.save_pretrained(OUTPUT_BASE)
    print(f"Saved in {OUTPUT_BASE}\n")


if __name__ == "__main__":
    if SAVE_LORA_ONLY:
        save_lora_only()
        torch.cuda.empty_cache()

    if SAVE_MERGED:
        save_merged()
        torch.cuda.empty_cache()

    if SAVE_BASE:
        save_base()
        torch.cuda.empty_cache()