import os

def delete_short_files(directory, min_word_count=5):
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file
        if os.path.isfile(file_path):
            # Open the file and read its content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                word_count = len(content.split())
                
                # Check if the word count is less than the minimum
                if word_count < min_word_count:
                    try:
                        print(f"Deleting file: {filename} (Word count: {word_count})")
                        os.remove(file_path)  # Delete the file
                    except PermissionError:
                        print(f"Cannot delete file: {filename}. It may be open in another program.")

if __name__ == "__main__":
    directory_to_check = "E:/gpt-2-simple-master/File_blocks"  # Change this to your directory
    delete_short_files(directory_to_check)
