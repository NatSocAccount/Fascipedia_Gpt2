from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./gpt2-italy-model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
input_text = "The history of Italy begins with"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate the next 100 tokens
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
