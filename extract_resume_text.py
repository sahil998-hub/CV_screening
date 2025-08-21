import fitz 

def extract_clean_resume_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text.strip() + "\n"

        clean_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )

        return clean_text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

if __name__ == "__main__":
    pdf_path = "resume.pdf" 
    resume_text = extract_clean_resume_text(pdf_path)

    print("=== CLEAN RESUME TEXT ===\n")
    print(resume_text)
