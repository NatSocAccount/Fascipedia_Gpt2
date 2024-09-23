import os
import nltk
import torch
from nltk import pos_tag, word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

stop_words = set(["a", "an", "the", "in", "on", "of", "for"])
print("Script started")

# Load the tokenizer and model
def setup_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model

def extract_nouns_and_verbs(text):
    print("Extracting nouns and verbs...")
    words = word_tokenize(text)
    tagged = pos_tag(words)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    verbs = [word for word, pos in tagged if pos.startswith('VB')]
    return nouns, verbs

def refine_text(generated_answer):
    sentences = nltk.sent_tokenize(generated_answer)
    one_sentence_summary = sentences[0] if sentences else ""
    one_paragraph_summary = "\n".join(sentences[:3]) if len(sentences) >= 3 else generated_answer

    def remove_repetitions(text):
        words = text.split()
        prev_word = None
        refined_words = []
        for word in words:
            if word != prev_word:
                refined_words.append(word)
            prev_word = word
        return ' '.join(refined_words)

    one_sentence_summary = remove_repetitions(one_sentence_summary)
    one_paragraph_summary = remove_repetitions(one_paragraph_summary)

    return one_sentence_summary, one_paragraph_summary

def generate_answers(input_nouns, search_content, tokenizer, model):
    import torch  # Ensure PyTorch is imported
    generated_answers = []

    # Ensure the padding token and EOS token are distinct
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id + 1  # Assign a unique pad_token_id

    # Limit the number of context lines to prevent long sequences
    context_lines = search_content.split('\n')
    context_lines = context_lines[-10:]  # Use last 10 lines as an example
    search_content = "\n".join(context_lines)

    for noun in input_nouns:
        context_lines = [line for line in search_content.split('\n') if noun.lower() in line.lower()]
        
        if context_lines:
            context = "\n".join(context_lines)
            print(f"Generating response for '{noun}' with context length {len(context)}...")

            # Tokenize input and create attention mask
            input_ids = tokenizer.encode(f"Question: {noun}\nContext: {context}\nAnswer:", return_tensors='pt')

            # Ensure input_ids is a tensor and create attention mask
            input_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
            attention_mask = torch.ones_like(input_ids)  # Create an attention mask of ones (same shape as input_ids)

            # Check and truncate input if too long
            if input_ids.size(1) > 1024:  # Adjust if needed
                input_ids = input_ids[:, -1024:]  # Keep last 1024 tokens
                attention_mask = attention_mask[:, -1024:]  # Adjust attention mask accordingly

            # Generate answer
            output = model.generate(
                input_ids,
                max_new_tokens=50,  # Adjust token length as necessary
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,    # Lower temperature for less random output
                top_k=50,           # Consider top 50 tokens in each step
                top_p=0.9           # Consider the top 90% probability mass for sampling
            )


            # Decode generated answer
            generated_answer = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Refine the generated answer
            one_sentence_summary, one_paragraph_summary = refine_text(generated_answer)

            # Append the refined answer
            generated_answers.append({
                'noun': noun,
                'answer': generated_answer,
                'one_sentence_summary': one_sentence_summary,
                'one_paragraph_summary': one_paragraph_summary
            })
        else:
            print(f"No context found for noun: {noun}")
            generated_answers.append({
                'noun': noun,
                'answer': "No relevant context found.",
                'one_sentence_summary': "",
                'one_paragraph_summary': ""
            })

    return generated_answers



def find_relevant_files(search_terms, directory):
    relevant_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(term.lower() in file.lower() for term in search_terms):
                relevant_files.append(os.path.join(root, file))
    return relevant_files

def main():
    print("Running...")
    input_question = "Nazi Hitler action"
    search_directory = "E:/gpt-2-simple-master/File_blocks/"
    output_folder = "GeneratedAnswers"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    input_nouns, input_verbs = extract_nouns_and_verbs(input_question)

    # Combine nouns and verbs for searching
    search_terms = input_nouns + input_verbs

    # Find relevant files
    search_files = find_relevant_files(search_terms, search_directory)

    # Read content from relevant files
    search_content = ""
    for file in search_files:
        with open(file, 'r', encoding='utf-8') as f:
            search_content += f.read() + "\n"  # Combine content

    tokenizer, model = setup_model(model_name="gpt2")
    generated_answers = generate_answers(input_nouns, search_content, tokenizer, model)

    # Saving answers in a dynamic folder system
    for answer_data in generated_answers:
        noun = answer_data['noun']
        answer = answer_data['answer']
        one_sentence_summary = answer_data['one_sentence_summary']
        one_paragraph_summary = answer_data['one_paragraph_summary']

        noun_folder = os.path.join(output_folder, noun.lower())
        os.makedirs(noun_folder, exist_ok=True)

        with open(os.path.join(noun_folder, f"{noun}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Answer for {noun}:\n{answer}\n\nOne Sentence Summary:\n{one_sentence_summary}\n\nOne Paragraph Summary:\n{one_paragraph_summary}\n")

if __name__ == "__main__":
    main()
