import os
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def chunk_text_by_sentences(text, min_sentences=2, max_sentences=10):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= min_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:  # Append any leftover sentences
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_nouns_and_verbs(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    verbs = [word for word, pos in tagged if pos.startswith('VB')]
    return nouns, verbs

def main():
    input_file_path = "E:/gpt-2-simple-master/Text/database_folder/FasciPediaarticles.txt"
    output_directory = "E:/gpt-2-simple-master/File_blocks"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into paragraphs
    paragraphs = content.split('\n\n')  # Assuming paragraphs are separated by double newlines

    for i, paragraph in enumerate(paragraphs):
        # Chunk the paragraph into smaller text blocks
        text_chunks = chunk_text_by_sentences(paragraph)

        for chunk in text_chunks:
            nouns, verbs = extract_nouns_and_verbs(chunk)
            all_words = nouns + verbs
            
            # Get the 10 most common nouns and verbs
            common_words = Counter(all_words).most_common(10)
            file_name = '_'.join([word[0] for word in common_words if word[0].isalpha()])  # Join common words
            
            if not file_name:  # Skip if no common words found
                continue

            # Limit the file name length to avoid issues with file systems
            file_name = file_name[:500] + '.txt'  # Keep it to 50 characters max

            # Write the chunk to a new file named after the common words
            output_file_path = os.path.join(output_directory, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(chunk)

            print(f"Saved: {output_file_path}")

if __name__ == "__main__":
    main()
