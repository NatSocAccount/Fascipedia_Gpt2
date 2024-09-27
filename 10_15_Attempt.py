import os
import gpt_2_simple as gpt2
import enchant
import nltk
from nltk.tokenize import sent_tokenize

stop_words = set(["a", "an", "the", "in", "on", "of", "for"])
print("Script started")

def extract_nouns(text):
    print("nouns")
    words = text.split()
    nouns = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    return nouns

def refine_text(generated_answer, output_file):
    # Tokenize the generated answer into sentences and paragraphs using NLTK
    sentences = nltk.sent_tokenize(generated_answer)
    paragraphs = generated_answer.split('\n\n')  # Assuming paragraphs are separated by double line breaks

    # Select the first sentence and the first paragraph (adjust as needed)
    one_sentence_summary = sentences[0] if sentences else ""
    one_paragraph_summary = paragraphs[0] if paragraphs else ""

    # Use PyEnchant to correct grammar and spelling errors in the selected text
    def correct_text(text):
        english_dict = enchant.Dict("en_US")
        corrected_text = " ".join(english_dict.suggest(word)[0] if not english_dict.check(word) else word for word in text.split())
        return corrected_text

    # Correct grammar and spelling in the selected summaries
    one_sentence_summary = correct_text(one_sentence_summary)
    one_paragraph_summary = correct_text(one_paragraph_summary)

    # Remove repetitive phrases (use a more advanced algorithm if needed)
    def remove_repetitions(text):
        words = text.split()
        prev_word = None
        refined_words = []
        for word in words:
            if word != prev_word:
                refined_words.append(word)
            prev_word = word
        return ' '.join(refined_words)

    # Remove repetitions in the selected summaries
    one_sentence_summary = remove_repetitions(one_sentence_summary)
    one_paragraph_summary = remove_repetitions(one_paragraph_summary)

    # Save the summaries to their respective files
    with open(output_file, "a", encoding="utf-8") as summary_file:
        summary_file.write("\n\nOne Sentence Summary:\n")
        summary_file.write(one_sentence_summary)
        summary_file.write("\n\nOne Paragraph Summary:\n")
        summary_file.write(one_paragraph_summary)

    return one_sentence_summary, one_paragraph_summary

def Generate_loki(input_nouns, search_content, output_folder, model_name="774M"):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    generated_answers = []

    for noun in input_nouns:
        # Initialize a separate output file for each summary
        output_file = os.path.join(output_folder, f"{noun}.txt")

        if noun not in search_content:
            print(f"No results for the noun: {noun}")
            user_input = input(f"Enter text for '{noun}.txt': ")
            with open(output_file, "w", encoding="utf-8") as loki_output:
                loki_output.write(user_input)
        else:
            context = []
            for line in search_content.split('\n'):
                if noun.lower() in line.lower():
                    context.append(line)

            with open(output_file, 'w', encoding='utf-8') as loki_output:
                loki_output.write('\n'.join(context))

            if not os.path.isdir(os.path.join("models", model_name)):
                print(f"Downloading {model_name} model...")
                gpt2.download_gpt2(model_name=model_name)

            sess = gpt2.start_tf_sess()
            gpt2.finetune(sess, output_file, model_name=model_name, steps=15)
            print("Answer is generated")

            generated_answer = gpt2.generate(sess, return_as_list=True)[0]

            refined_answer = refine_text(generated_answer)
            one_sentence_summary, one_paragraph_summary = refine_text(generated_answer, "output_summary.txt")
                
            with open(output_file, "w", encoding="utf-8") as answer_output:
                answer_output.write(refined_answer)
                answer_output.write(one_sentence_summary)
                answer_output.write(one_paragraph_summary)

            generated_answers.append(refined_answer)

    return generated_answers


def main():
    print("running1")
    input_question = "How can I fix my broken arm?"
    search_file = "E:\\gpt-2-simple-master\\OdinCronos3.txt"  # Replace with the path to your search file
    output_folder = f"BrokeArm"  # Output folder named "Apple"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(search_file, 'r', encoding='utf-8') as file:
        search_content = file.read()

    input_nouns = extract_nouns(input_question)
    model_name = "774M"

    print("Generating Loki")
    generated_answers = Generate_loki(input_nouns, search_content, output_folder, model_name=model_name)

    print("Answers generated:")
    for noun, answer in zip(input_nouns, generated_answers):
        print(f"Noun: {noun}")
        print(f"Answer: {answer}")
        print("===")

if __name__ == "__main__":
    main()
   