import os
import nltk
import torch
import random
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

stop_words = set(["a", "an", "the", "in", "on", "of", "for"])
print("Script started")

def setup_model(model_name="gpt2"):
    """Load the tokenizer and model."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def extract_nouns_and_verbs(text):
    """Extract nouns and verbs from a given text."""
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    # Filter nouns and verbs
    nouns_and_verbs = [word for word, tag in tagged_words if tag.startswith('N') or tag.startswith('V')]
    return nouns_and_verbs

def summarize_text(text, num_sentences=2):
    """Summarize text using sentence tokenization."""
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:num_sentences])  # Join the first few sentences

def truncate_text(text, max_length, tokenizer):
    """Truncate text to a maximum length in tokens."""
    if not text:  # Check if the text is empty
        return ""
    
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def summarize_in_chunks(text, max_words=200):
    """Summarize text into manageable chunks while respecting a word limit."""
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    summary = []
    word_count = 0

    for sentence in sentences:
        # Check the current word count and the words in the new sentence
        sentence_word_count = len(sentence.split())
        if word_count + sentence_word_count > max_words:
            break  # Stop if adding this sentence exceeds the word limit
        
        summary.append(sentence)
        word_count += sentence_word_count

    return " ".join(summary)

def calculate_similarity(content1, content2):
    """Calculate cosine similarity between two pieces of content."""
    vectorizer = CountVectorizer().fit_transform([content1, content2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]  # Cosine similarity between the two texts

def rank_files(search_terms, search_directory):
    """Rank files based on similarity to search terms and return top 100."""
    ranked_files = []
    
    for filename in os.listdir(search_directory):
        file_path = os.path.join(search_directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                similarity_score = calculate_similarity(" ".join(search_terms), file_content)
                ranked_files.append((similarity_score, file_path, file_content))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Sort files based on similarity scores (highest first) and keep the top 100
    ranked_files.sort(key=lambda x: x[0], reverse=True)
    return ranked_files[:50]  # Keep only the top 100 files

def generate_answer(query, combined_summary, tokenizer, model):
    """Generate an answer based on the query and combined summary."""
    max_length = 1024  # Max length for GPT-2
    truncated_summary = truncate_text(combined_summary, max_length, tokenizer)

    if not truncated_summary:  # Check if the summary is empty
        print("Warning: Truncated summary is empty. Cannot generate answers.")
        return None  # Changed to None for clarity

    input_ids = tokenizer.encode(truncated_summary, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Ensure the input is within the model's capacity
    if input_ids.size(1) == 0 or input_ids.size(1) > max_length:
        print("Invalid input IDs size. Check input text length.")
        return None  # Changed to None for clarity

    try:
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        one_sentence_summary = answer.split('.')[0].strip()  # Clean up
        one_paragraph_summary = answer.strip()  # Clean up

        return {
            'question': query,
            'answer': answer,
            'one_sentence_summary': one_sentence_summary,
            'one_paragraph_summary': one_paragraph_summary
        }
    except Exception as e:
        print(f"Error during generation: {e}")
        return None  # Changed to None for clarity

    
def clean_generated_text(text):
    """Clean up the generated text to remove unwanted characters."""
    # Remove extra newlines and spaces
    cleaned_text = text.replace("\n", " ").strip()
    # Optionally, you can implement more sophisticated cleaning here
    return cleaned_text

def main():
    print("Running...")
    input_question = "Adolf Hitler waffen SS"
    search_directory = "E:/gpt-2-simple-master/File_blocks/"
    output_folder = "GeneratedAnswers"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    query = input_question
    important_keywords = extract_nouns_and_verbs(input_question)
    relevant_files_content = []

    if important_keywords:
        ranked_files = rank_files(important_keywords, search_directory)

        # Read and summarize content from the top ranked files
        for _, file_path, file_content in ranked_files:
            # Summarize the content in chunks
            summary = summarize_in_chunks(file_content)
            relevant_files_content.append(summary)
    else:
        print("No keywords extracted from the original question.")

    combined_summary = "\n".join(relevant_files_content)
    tokenizer, model = setup_model(model_name="gpt2")

    # Generate answers while ensuring input is within limits
    generated_answer = generate_answer(query, combined_summary, tokenizer, model)

    if generated_answer is not None:
        cleaned_answers = clean_generated_text(generated_answer['answer'])  # Clean the main answer
        one_sentence_summary = generated_answer['one_sentence_summary']
        one_paragraph_summary = generated_answer['one_paragraph_summary']
        
        # Improve answer similarity
        refined_answer, refined_sentence_summary, refined_paragraph_summary = improve_answer_similarity(
            query, cleaned_answers, one_sentence_summary, one_paragraph_summary
        )

        # Output only the refined summaries
        print("Refined Sentence Summary:")
        print(refined_sentence_summary)
        print("\nRefined Paragraph Summary:")
        print(refined_paragraph_summary)
    else:
        print("No valid answer generated.")


def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]  # Cosine similarity between the two texts

def refine_text(generated_answer):
    """Refine the generated answer into a more concise form."""
    # For simplicity, we can summarize it again or just take the first few sentences.
    return summarize_text(generated_answer, num_sentences=2), generated_answer  # Return both summaries

def improve_answer_similarity(original_question, generated_answer, one_sentence_summary, one_paragraph_summary):
    """
    This function attempts to improve the similarity between the generated answer and the original question
    by either modifying the answer or re-generating it.
    """
    
    important_keywords = extract_nouns_and_verbs(original_question)
    
    # Step 1: Check if important keywords are missing
    missing_keywords = [keyword for keyword in important_keywords if keyword not in generated_answer]

    # If missing keywords, try to add them to the one_sentence_summary
    if missing_keywords:
        print(f"Missing keywords: {missing_keywords}. Attempting to refine the answer.")
        
        # Append missing keywords to summaries
        refined_sentence_summary = f"{one_sentence_summary}. Keywords: {' '.join(missing_keywords)}."
        refined_paragraph_summary = f"{one_paragraph_summary}. Additionally, consider: {' '.join(missing_keywords)}."
        
        return generated_answer, refined_sentence_summary, refined_paragraph_summary

    # Step 2: Re-generate the answer if necessary
    similarity_score = calculate_cosine_similarity(original_question, generated_answer)
    if similarity_score < 0.5:
        print(f"Cosine similarity is too low ({similarity_score:.2f}), re-generating the answer.")
        
        # Optionally, you could refine the query or re-prompt the model for a new answer
        refined_answer = generate_answer(original_question)  # Placeholder for model re-generation logic
        
        # You would then re-summarize the refined answer
        refined_sentence_summary, refined_paragraph_summary = refine_text(refined_answer)
        
        return refined_answer, refined_sentence_summary, refined_paragraph_summary
    
    return generated_answer, one_sentence_summary, one_paragraph_summary

if __name__ == "__main__":
    main()
