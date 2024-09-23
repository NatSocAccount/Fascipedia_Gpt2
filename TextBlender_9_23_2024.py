import os
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

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
        nouns, verbs = extract_nouns_and_verbs(paragraph)
        all_words = nouns + verbs
        
        # Get the 10 most common nouns and verbs
        common_words = Counter(all_words).most_common(10)
        file_name = '_'.join([word[0] for word in common_words if word[0].isalpha()])  # Join common words
        if not file_name:  # Skip if no common words found
            continue

        # Limit the file name length to avoid issues with file systems
        file_name = file_name[:50] + '.txt'  # Keep it to 50 characters max

        # Write the paragraph to a new file named after the common words
        output_file_path = os.path.join(output_directory, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(paragraph)

        print(f"Saved: {output_file_path}")

if __name__ == "__main__":
    main()
