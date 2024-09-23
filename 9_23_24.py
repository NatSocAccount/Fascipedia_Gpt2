import os
import nltk
from nltk import pos_tag, word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')  # Ensure this is downloaded for sentence tokenization
nltk.download('averaged_perceptron_tagger_eng')  # Ensure this is downloaded for POS tagging

stop_words = set(["a", "an", "the", "in", "on", "of", "for"])
print("Script started")


# Load the tokenizer and model
def setup_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model


def extract_nouns(text):
    print("Extracting nouns...")
    words = word_tokenize(text)
    tagged = pos_tag(words)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]  # Extract only nouns
    return nouns

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

# Example of how to set up your model and tokenizer (if not already done)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_answers(input_nouns, search_content, tokenizer, model):
    generated_answers = []

    # Loop through each noun
    for noun in input_nouns:
        context_lines = [line for line in search_content.split('\n') if noun.lower() in line.lower()]
        
        if context_lines:
            context = "\n".join(context_lines)
            print(f"Generating response for '{noun}'...")

            # Encode input and truncate if necessary
            input_ids = tokenizer.encode(f"Question: {noun}\nContext: {context}\nAnswer:", return_tensors='pt')

            # Truncate input if it's too long
            if input_ids.size(1) > 512:  # Adjust this based on model capabilities
                input_ids = input_ids[:, -512:]  # Keep only the last 512 tokens

            # Generate answer
            output = model.generate(input_ids, max_new_tokens=100)  # Use max_new_tokens

            # Decode and refine the generated answer
            generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)

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


def main():
    print("Running...")
    input_question = "Adolf Hitler"
    search_file = "E:\gpt-2-simple-master\Text\database_folder\FasciPediaarticles.txt"
    output_folder = "GeneratedAnswers"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(search_file, 'r', encoding='utf-8') as file:
        search_content = file.read()

    input_nouns = extract_nouns(input_question)

    tokenizer, model = setup_model(model_name="gpt2")
    generated_answers = generate_answers(input_nouns, search_content, tokenizer, model)

    for answer_data in generated_answers:
        noun = answer_data['noun']
        answer = answer_data['answer']
        one_sentence_summary = answer_data['one_sentence_summary']
        one_paragraph_summary = answer_data['one_paragraph_summary']

        print(f"Noun: {noun}")
        print(f"Answer: {answer}")
        print(f"One Sentence Summary: {one_sentence_summary}")
        print(f"One Paragraph Summary: {one_paragraph_summary}")
        print("===")

        with open(os.path.join(output_folder, f"{noun}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Answer for {noun}:\n{answer}\n\nOne Sentence Summary:\n{one_sentence_summary}\n\nOne Paragraph Summary:\n{one_paragraph_summary}\n")

if __name__ == "__main__":
    main()
