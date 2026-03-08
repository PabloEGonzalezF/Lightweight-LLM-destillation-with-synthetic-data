
#For using qwen 3.5 use: pip install git+https://github.com/huggingface/transformers.git

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("unsloth/Qwen3.5-4B")
model = AutoModelForImageTextToText.from_pretrained("unsloth/Qwen3.5-4B").to(device)

#For inference
model = torch.compile(model)

PROMPTS_FILE = "prompts.json"
OUTPUT_FILE = "teacher_responses.json"
#MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS = 4096

def generate_response(model, processor, prompt: str) -> str:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=False,
      return_dict=True,
      enable_thinking = False,
      enable_reasoning = False,
      return_tensors="pt",
    )

    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

if __name__ == "__main__":

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print("Teacher model loaded\n")

    results = []
    for item in prompts:
        prompt_id = item["id"]
        prompt_text = item["prompt"]
        print(f"[{prompt_id}/{len(prompts)}] Responses")

        response = generate_response(model, processor, prompt_text)

        #print(response)

        results.append({
            "id": prompt_id,
            "prompt": prompt_text,
            "response": response,
        })

        print(f"Done\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


    print(f"All responses saved to {OUTPUT_FILE}")
