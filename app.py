import streamlit as st
import nltk
import re
import pickle
import fitz 

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

def cleanResume(resume_text):
    cleanTxt = re.sub('http\S+', ' ', resume_text)
    cleanTxt = re.sub('@\S+', ' ', cleanTxt)
    cleanTxt = re.sub('#\S+', ' ', cleanTxt)
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    return cleanTxt.strip()

def extract_text_from_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text.strip() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def main():
    st.set_page_config(page_title="Resume Screening App", layout="centered")
    st.title("📄 Resume Category Predictor")

    uploaded_file = st.file_uploader("Upload a Resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        resume_bytes = uploaded_file.read()

        if uploaded_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_bytes)
        else:
            try:
                resume_text = resume_bytes.decode("utf-8")
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode("latin-1")

        if not resume_text.strip():
            st.warning("No text could be extracted from the file. Please try a different resume.")
            return

        with st.spinner("🔍 Analyzing resume..."):
            cleaned_resume = cleanResume(resume_text)
            transformed_resume = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(transformed_resume)[0]
            predicted_category = le.inverse_transform([prediction_id])[0]

        st.success("✅ Prediction complete!")
        st.markdown(f"### 📌 Predicted Category: `{predicted_category}`")

if __name__ == "__main__":
    main()
