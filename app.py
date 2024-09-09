from flask import Flask, render_template, request, jsonify
import PyPDF2
from job_recommender_project import GetJobRecommendation

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume_and_skills():
    resume = request.files['resume']
    resume_path = 'tempResume.pdf'
    resume.save(resume_path)

    with open(resume_path, 'rb') as resume_file:
        resume_reader = PyPDF2.PdfReader(resume_file)
        resume_text = ""
        for page_num in range(len(resume_reader.pages)):
            resume_text += resume_reader.pages[page_num].extract_text()

    skills = extract_skills(resume_text)
    job_recommendations = GetJobRecommendation(skills)
    
    # Return the job recommendations directly if they are already in list format
    return jsonify(job_recommendations)

def extract_skills(text):
    skills_keywords = ["Python", "Java", "Machine Learning", "TensorFlow", "SQL", "Flask", "Django", "HTML", "CSS", "JavaScript", "Git", "GitHub", "AWS", "MySQL", "Numpy", "Pandas", "Matplotlib"]
    skills = []
    for keyword in skills_keywords:
        if keyword in text:
            skills.append(keyword)
    return skills

if __name__ == '__main__':
    app.run(debug=True)
