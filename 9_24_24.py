import os
import nltk
import torch
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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
    nouns_and_verbs = [word for word, tag in tagged_words if tag.startswith('N') or tag.startswith('V')]
    return nouns_and_verbs

def summarize_in_chunks(text, max_words=200):
    """Summarize text into manageable chunks while respecting a word limit."""
    sentences = sent_tokenize(text)
    summary = []
    word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if word_count + sentence_word_count > max_words:
            break
        
        summary.append(sentence)
        word_count += sentence_word_count

    return " ".join(summary)

def calculate_similarity(content1, content2):
    """Calculate cosine similarity between two pieces of content."""
    vectorizer = CountVectorizer().fit_transform([content1, content2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def rank_files(search_terms, search_directory):
    """Rank files based on similarity to search terms and return top 50."""
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

    ranked_files.sort(key=lambda x: x[0], reverse=True)
    return ranked_files[:50]  # Keep only the top 50 files

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts."""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]  # Cosine similarity between the two texts

def refine_text(generated_answer):
    """Refine the generated answer into a more concise form."""
    # For simplicity, we can summarize it again or just take the first few sentences.
    return summarize_in_chunks(generated_answer, max_words=50), generated_answer  # Return both summaries

def improve_answer_similarity(original_question, generated_answer, one_sentence_summary, one_paragraph_summary, combined_summary, tokenizer, model):
    """
    Improve the similarity between the generated answer and the original question
    by modifying the answer or re-generating it.
    """
    
    important_keywords = extract_nouns_and_verbs(original_question)
    
    # Step 1: Check if important keywords are missing
    missing_keywords = [keyword for keyword in important_keywords if keyword not in generated_answer]

    # If missing keywords, try to add them to the summaries
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
        
        # Call generate_answer with all necessary parameters
        refined_answer = generate_answer(original_question, combined_summary, tokenizer, model)
        
        # You would then re-summarize the refined answer
        refined_sentence_summary, refined_paragraph_summary = refine_text(refined_answer['answer'])
        
        return refined_answer['answer'], refined_sentence_summary, refined_paragraph_summary
    
    return generated_answer, one_sentence_summary, one_paragraph_summary


def generate_answer(query, combined_summary, tokenizer, model):
    """Generate an answer based on the query and combined summary, handling long inputs."""
    max_length = 1024  # Max length for GPT-2
    chunk_size = 512    # Max number of tokens for each chunk
    responses = []

    # Split combined summary into manageable chunks at the text level
    sentences = sent_tokenize(combined_summary)
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length > chunk_size:
            # Process the current chunk
            chunk_text = " ".join(current_chunk)
            prompt_text = f"Based on the following information, provide a one-sentence summary and a five-sentence summary:\n{chunk_text}\n\nOne-sentence summary:\n"
            input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

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
                responses.append(answer.strip())
            except Exception as e:
                print(f"Error during generation: {e}")
                return None
            
            # Reset for the next chunk
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Process any remaining sentences in the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        prompt_text = f"Based on the following information, provide a one-sentence summary and a five-sentence summary:\n{chunk_text}\n\nOne-sentence summary:\n"
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

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
            responses.append(answer.strip())
        except Exception as e:
            print(f"Error during generation: {e}")
            return None

    # Combine responses from all chunks into a single response
    combined_response = " ".join(responses).replace("\n", " ").strip()
    
    return {
        'question': query,
        'answer': combined_response
    }

def clean_generated_text(text):
    """Clean up the generated text to remove unwanted characters."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def sanitize_filename(filename):
    # Replace any character that is not a letter, number, or underscore with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

   
def main():
    print("Running...")
    input_question = "Where is Italy?"
    search_directory = "E:/gpt-2-simple-master/File_blocks/"
    output_folder = "GeneratedAnswers"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    query = input_question
    important_keywords = extract_nouns_and_verbs(input_question)
    relevant_files_content = []

    if important_keywords:
        ranked_files = rank_files(important_keywords, search_directory)

        for _, file_path, file_content in ranked_files:
            summary = summarize_in_chunks(file_content)
            relevant_files_content.append(summary)
    else:
        print("No keywords extracted from the original question.")

    combined_summary = "\n".join(relevant_files_content)
    tokenizer, model = setup_model(model_name="gpt2")

    generated_answer = generate_answer(query, combined_summary, tokenizer, model)

    if generated_answer is not None:
        # Refine the generated answer and check for similarity
        one_sentence_summary, one_paragraph_summary = refine_text(generated_answer['answer'])
        improved_answer, refined_sentence_summary, refined_paragraph_summary = improve_answer_similarity(
            input_question,
            generated_answer['answer'],
            one_sentence_summary,
            one_paragraph_summary,
            combined_summary,
            tokenizer,
            model
        )

        # Clean the text before saving
        improved_answer = clean_generated_text(improved_answer)

        # Sanitize the filename before saving the file
        output_file_name = f"generated_answer_{input_question}.txt"
        sanitized_file_name = sanitize_filename(output_file_name)  # Make sure filename is valid
        output_file_path = os.path.join(output_folder, sanitized_file_name)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(improved_answer)

        print(f"Generated answer saved to {output_file_path}")
    else:
        print("Failed to generate an answer.")

if __name__ == "__main__":
    main()

