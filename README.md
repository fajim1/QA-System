# Question Answering System

This repository contains a simple question answering system using the Hugging Face's Transformers library. It fetches large documents from a given URL or PDF and processes them to find the best answers to a given question.

## Requirements

- Python 3.6 or later
- Transformers
- Requests
- BeautifulSoup4
- PyPDF2 (for processing PDFs)
- tqdm (for progress bar)
- PyMuPDF (for extracting text from PDFs stored in Amazon S3)

## Usage

1. Clone this repository.
2. Install the required packages.
3. Run the provided Jupyter Notebook or use the code in your own project.

## Example

An example using a Wikipedia page as the source document is provided in the Jupyter Notebook. The question answering system processes the page and returns the top 5 answers along with the corresponding snippets from the document.

## License

MIT License

