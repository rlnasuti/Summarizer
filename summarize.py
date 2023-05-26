from transformers import pipeline
import PyPDF2

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_number in range(len(pdf_reader.pages)):
                print(f"Processing page {page_number + 1}...")
                page = pdf_reader.pages[page_number]
                page_text = page.extract_text()
                text += page_text if page_text is not None else ""
            return text
    except Exception as e:
        return str(e)
    
def summarize_text(text, max_length=150, summarizer=None):
    try:
        summary = summarizer(text, max_length=max_length, do_sample=True)
        return summary[0]['summary_text']
    except Exception as e:
        return str(e)

# Main guard
if __name__ == '__main__':
    # Specify the path to your PDF file
    pdf_file_path = '/Users/robertnasuti/Downloads/gpt-4-system-card.pdf'
    pdf_text = extract_text_from_pdf(pdf_file_path)
    
    # Initialize the summarizer
    summarizer = pipeline("summarization", model="t5-base")
    
    # Specify the output file path
    output_file_path = '/Users/robertnasuti/Desktop/Dev/Summarizer/gpt-4-system-card-summary.txt'
    
    # Split text into smaller chunks for summarization
    chunk_size = 1024
    text_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
    
    # Summarize each chunk and write to the output file
    with open(output_file_path, 'w') as output_file:
        for i, chunk in enumerate(text_chunks):
            print(f"Summarizing chunk {i + 1}/{len(text_chunks)}...")
            summary = summarize_text(chunk, max_length=1500, summarizer=summarizer)
            output_file.write(summary + '\n')
    
    print(f"Summary saved to {output_file_path}")
