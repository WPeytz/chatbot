from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./gpt2-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned")

# Function to generate responses
def chatbot_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = input_ids != tokenizer.pad_token_id  # Create attention mask

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Pass attention mask
        max_length=100,  # Generate longer responses
        num_return_sequences=1,
        temperature=1.0,  # Increase creativity
        top_k=50,
        top_p=0.95,
        do_sample=True,  # Enable sampling
        repetition_penalty=1.2,  # Penalize repetitive text
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Test the chatbot
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(prompt))