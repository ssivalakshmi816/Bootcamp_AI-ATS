import gradio as gr
from sentence_transformers import SentenceTransformer, util
import re
import pdfplumber

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Extract text from PDF ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file.name) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# --- Keyword extraction ---
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in set(words) if len(w) > 3]

# --- Main logic ---
def analyze(pdf_file, jd):
    if pdf_file is None or not jd.strip():
        return "⚠️ Please upload a resume PDF and enter a job description."

    # Extract resume text
    resume = extract_text_from_pdf(pdf_file)

    # Keywords
    resume_words = extract_keywords(resume)
    jd_words = extract_keywords(jd)

    matched = set(resume_words) & set(jd_words)
    missing = set(jd_words) - set(resume_words)

    keyword_score = len(matched) / max(len(jd_words), 1) * 100

    # Semantic similarity
    emb1 = model.encode(resume)
    emb2 = model.encode(jd)
    similarity = util.cos_sim(emb1, emb2).item() * 100

    # Final score
    final_score = int(0.6 * keyword_score + 0.4 * similarity)

    # Suggestions
    suggestions = []
    if missing:
        suggestions.append(f"🔴 Add keywords: {', '.join(list(missing)[:10])}")
    else:
        suggestions.append("🟢 Strong keyword match!")

    if similarity < 60:
        suggestions.append("⚠️ Improve alignment with job description wording.")

    # Output
    return f"""
🎯 ATS Score: {final_score}/100
📊 Breakdown:
- Keyword Match: {int(keyword_score)}%
- Semantic Similarity: {int(similarity)}%
🟢 Matched Keywords:
{', '.join(list(matched)[:10]) or 'None'}
🔴 Missing Keywords:
{', '.join(list(missing)[:10]) or 'None'}
💡 Suggestions:
{chr(10).join(suggestions)}
"""

# --- UI (stable version) ---
with gr.Blocks() as demo:
    gr.Markdown("## ⚡ ATS Resume Analyzer")
    gr.Markdown("Upload your resume (PDF) and paste the job description.")

    pdf_input = gr.File(label="📄 Upload Resume (PDF)")
    jd_input = gr.Textbox(
        label="🧾 Job Description",
        lines=10,
        placeholder="Paste job description here..."
    )

    analyze_btn = gr.Button("Analyze")

    output = gr.Textbox(label="Result", lines=15)

    analyze_btn.click(fn=analyze, inputs=[pdf_input, jd_input], outputs=output)

# --- Launch ---
if __name__ == "__main__":
    demo.launch()
