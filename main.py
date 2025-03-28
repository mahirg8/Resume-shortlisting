from flask import Flask, request, render_template
import os
import docx2txt
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

# Extract text from PDF using pdfplumber
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return preprocess_text(text)

# Extract text from DOCX
def extract_text_from_docx(file_path):
    return preprocess_text(docx2txt.process(file_path))

# Extract text from TXT
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return preprocess_text(file.read())

# Determine file type and extract text
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        resumes = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resumes.append(extract_text(filename))

        if not resumes or not job_description:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        # Preprocess job description
        job_description = preprocess_text(job_description)

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        vectors = vectorizer.fit_transform([job_description] + resumes).toarray()

        # Compute cosine similarity
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Print similarity scores in console
        print("\nResume Similarity Scores:")
        for i, sim in enumerate(similarities):
            print(f"Resume {i+1}: {sim:.2f}")

        # Set similarity threshold for accuracy
        threshold = 0.3  # Adjust this value as needed
        matching_resumes = sum(sim > threshold for sim in similarities)
        accuracy = (matching_resumes / len(resumes)) * 100 if resumes else 0

        # Print accuracy in console
        print(f"\nMatching Resumes: {matching_resumes}/{len(resumes)}")
        print(f"Calculated Accuracy: {accuracy:.2f}%\n")

        # Get top 5 resumes
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        return render_template('matchresume.html', 
                               message="Top matching resumes:", 
                               top_resumes=top_resumes, 
                               similarity_scores=similarity_scores)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
