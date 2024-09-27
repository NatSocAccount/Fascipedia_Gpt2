# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 00:08:00 2023

@author: A&M
"""

import gpt_2_simple as gpt2
import os

#from TextKey2 import extract_text_by_word_count  # Import the function

# Define your input and output file paths and keyword
input_file = "Odin.txt"
output_file = "Tyr3.txt"
word_count = 2000  # Change this to your desired keyword or input phrase! 

print("Processing input keywords")
# Call the function to extract and save text
#extract_text_by_word_count(input_file, output_file, word_count)

file_name = "OdinCronos2.txt"


model_name = "774M"
#model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


#file_name = "shakespeare.txt"
#if not os.path.isfile(file_name):
#	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#	data = requests.get(url)
#
#with open(file_name, 'w') as f:
    #f.write(data.text)

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1)   # steps is max number of training steps

gpt2.generate(sess)

output_text = gpt2.generate(sess, return_as_list=True)[0]  # Generate text and store it in a variable
print(output_text)  # Print the generated text

with open("Thor2.txt", "w", encoding="utf-8") as file:
    file.write(output_text)

print("Generated text saved to Thor2.txt")