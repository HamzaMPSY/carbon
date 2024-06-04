from flask import render_template, request, redirect, url_for
from app import app
from app.utils import preprocess_text, extract_top_sentences, tfidf_summary, wf_summary, extract_formulas, search_documents, read_file
import os


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        text = read_file(file_path)
        if not text:
            return redirect(url_for('index'))
        text = text.lower()
        preprocessed_text = preprocess_text(text)
        entities = extract_top_sentences(text, 5)
        tfidf_sum = tfidf_summary(text)
        wf_sum = wf_summary(text)
        formulas = extract_formulas(text)
        return render_template('result.html', text=text, preprocessed_text=preprocessed_text, entities=entities, tfidf_summary=tfidf_sum, wf_summary=wf_sum, formulas=formulas)
    return redirect(url_for('index'))


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        results = search_documents(query)
        return render_template('search_results.html', query=query, results=results)
    return render_template('index.html')


@app.route('/extract_formulas', methods=['POST'])
def extract_formulas_route():
    text = request.form['text']
    formulas = extract_formulas(text)
    return render_template('formulas.html', formulas=formulas)
