from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ------------------------
# Load your local model
# ------------------------
model_path = "gguf or any model "  # put path to your model folder or just put model in folder of code 

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",       # automatically uses GPU if available
    torch_dtype=torch.float16
)
model.eval()

# ------------------------
# Terminal chat loop
# ------------------------
print("=== Local LLM Chat ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input(">>> ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode output
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\n{response}\n")
