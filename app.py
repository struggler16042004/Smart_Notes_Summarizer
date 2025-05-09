from flask import Flask, request, render_template_string, send_file
from transformers import pipeline
from textblob import TextBlob
from PyPDF2 import PdfReader
from fpdf import FPDF
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Smart Notes Summarizer</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
  <style>
    body {
      background-color: #e8f0fe;
    }
    .container {
      max-width: 900px;
    }
    .card {
      border-radius: 12px;
    }
    textarea {
      resize: vertical;
      font-size: 1rem;
    }
    footer {
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-dark bg-primary mb-4">
  <div class="container-fluid">
    <span class="navbar-brand mb-0 h1"><i class="fas fa-book-reader"></i> Smart Notes Summarizer</span>
  </div>
</nav>

<div class="container">
  <div class="card shadow-sm mb-4">
    <div class="card-body">
      <form method="POST" enctype="multipart/form-data">
        <div class="mb-3">
          <label class="form-label"><strong>Paste Your Text</strong></label>
          <textarea name="text" rows="10" class="form-control" placeholder="Paste content here...">{{ request.form.text or '' }}</textarea>
        </div>
        <div class="mb-3">
          <label class="form-label"><strong>Or Upload a File</strong> (.txt or .pdf)</label>
          <input type="file" name="file" class="form-control">
        </div>
        <button type="submit" class="btn btn-primary w-100 btn-lg"><i class="fas fa-magic"></i> Summarize</button>
      </form>
    </div>
  </div>

  {% if summary %}
  <div class="card shadow-sm mb-4">
    <div class="card-header bg-success text-white">
      <i class="fas fa-lightbulb"></i> Summary
    </div>
    <div class="card-body">
      <p style="font-size: 1.1rem;">{{ summary }}</p>
    </div>
  </div>

  <div class="card shadow-sm mb-4">
    <div class="card-header bg-info text-white">
      <i class="fas fa-smile"></i> Sentiment Analysis
    </div>
    <div class="card-body">
      <p style="font-size: 1.1rem;">
        <strong>Polarity:</strong> {{ sentiment.polarity }}
        <span class="badge {% if sentiment.polarity > 0.1 %}bg-success{% elif sentiment.polarity < -0.1 %}bg-danger{% else %}bg-secondary{% endif %}">
          {% if sentiment.polarity > 0.1 %}
            Positive
          {% elif sentiment.polarity < -0.1 %}
            Negative
          {% else %}
            Neutral
          {% endif %}
        </span><br>
        <strong>Subjectivity:</strong> {{ sentiment.subjectivity }}
      </p>
    </div>
  </div>

  <a href="/download" class="btn btn-outline-success w-100 mb-4"><i class="fas fa-download"></i> Download Summary as PDF</a>
  {% endif %}

  <footer class="text-center text-muted mt-4">
    &copy; 2025 Smart NLP Summarizer | Built with Flask, Transformers & ❤️
  </footer>
</div>
</body>
</html>
'''


# File/text handling logic
def extract_text_from_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filename.endswith('.pdf'):
        with open(filepath, 'rb') as f:
            reader = PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif filename.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def summarize_text(text):
    max_input = 1024
    if len(text.split()) > max_input:
        text = " ".join(text.split()[:max_input])
    result = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return result[0]['summary_text']

def analyze_sentiment(text):
    return TextBlob(text).sentiment

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output("summary.pdf")

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = ""
    sentiment = None
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        input_text = request.form.get('text', '')

        if uploaded_file and uploaded_file.filename != "":
            input_text = extract_text_from_file(uploaded_file)

        if input_text.strip():
            summary = summarize_text(input_text)
            sentiment = analyze_sentiment(input_text)
            create_pdf(summary)

    return render_template_string(TEMPLATE, summary=summary, sentiment=sentiment)

@app.route('/download')
def download():
    return send_file("summary.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
