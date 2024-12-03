from transformers import GPT2Tokenizer

model_path = "gpt2-finetuned/checkpoint-3"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(model_path)