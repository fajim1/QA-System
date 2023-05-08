#Import required packages and libraries
#!pip install torch transformers requests beautifulsoup4 boto3 pdfplumber tqdm

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import time
from tqdm.auto import tqdm
import requests
from bs4 import BeautifulSoup

# Load the model and tokenizer from Hugging Face.
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# Set the query (question).
question = "Who is known as the father of computer?"


# import boto3
# import pdfplumber
# import io

# # AWS credentials
# aws_access_key_id = "your_aws_access_key_id"
# aws_secret_access_key = "your_aws_secret_access_key"

# # S3 bucket and object key
# bucket_name = "your_bucket_name"
# object_key = "your_object_key"

# # Initialize S3 client
# s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# # Get the PDF object from S3
# pdf_object = s3.get_object(Bucket=bucket_name, Key=object_key)

# # Read the PDF content into a BytesIO object
# pdf_content = io.BytesIO(pdf_object["Body"].read())

# # Extract text from the PDF
# document = ""
# with pdfplumber.open(pdf_content) as pdf:
#     for page in pdf.pages:
#         document += page.extract_text()

# # Now, you can use the `document` variable in the previous code snippet

# Fetch a large document from a URL, parse it using BeautifulSoup, and extract the text content.
url = "https://en.wikipedia.org/wiki/History_of_computing_hardware"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
document = soup.get_text()

# Set the parameters for chunk size and stride.
chunk_size = 384  # Tokens (Modify as needed)
stride = 192  # Tokens (Modify as needed)

# Split the document into overlapping chunks
tokens = tokenizer.encode(document)
document_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), stride)]

#Process each chunk with the model, and store the answers, scores, and chunks in a list.
all_answers = []
start_time = time.time()

# Loop through each chunk and process it using the model
for chunk in tqdm(document_chunks, desc="Processing chunks", unit="chunk"):
    # Prepare the inputs for the model using the tokenizer
    inputs = tokenizer(question, tokenizer.decode(chunk), return_tensors="pt", padding="max_length", max_length=chunk_size, truncation=True)
    
    # Get the model outputs
    outputs = model(**inputs)
    
    # Extract start and end logits from the model outputs
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    
    # Get the start and end indices of the answer
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    
    # Decode the answer from the input tokens
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx + 1]))
    
    # Add the answer, scores, and chunk to the list
    all_answers.append({"answer": answer, "start_score": start_scores[0][start_idx].item(), "end_score": end_scores[0][end_idx].item(), "chunk": chunk})


# Get the top 3 answers from all answers, and print the answers and the corresponding snippets.
top_answers = sorted(all_answers, key=lambda x: x["end_score"], reverse=True)[:3]

for idx, answer in enumerate(top_answers):
    print(f"Answer {idx + 1}: {answer['answer']}")
    snippet = tokenizer.decode(answer["chunk"])
    print(f"Snippet:\n{snippet}\n")

# Measure and print the time elapsed
elapsed_time = time.time() - start_time
print(f"Time elapsed: {elapsed_time:.2f} seconds")



import re

def create_safe_filename(filename: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", filename.strip())

# Create a safe filename from the question
filename = create_safe_filename(question) + ".txt"

# Write the answers and snippets to a .txt file using the question as the file name.
with open(filename, "w", encoding="utf-8") as file:
    
    file.write(f"Question: {question}\n\n")
    
    for idx, answer in enumerate(top_answers):
        file.write(f"Answer {idx + 1}: {answer['answer']}\n")
        snippet = tokenizer.decode(answer["chunk"])
        file.write(f"Snippet:\n{snippet}\n\n")

print(f"Answers and snippets written to '{filename}'.")

