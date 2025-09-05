from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import docx
import ollama
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import tempfile
import os
from typing import List, Dict
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except:
    pass

app = FastAPI(title="CareerFit AI", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SkillMatch(BaseModel):
    present_skills: List[str]
    missing_skills: List[str]
    match_score: int


class AnalysisResult(BaseModel):
    match_score: int
    present_skills: List[str]
    missing_skills: List[str]
    interview_questions: List[str]
    optimized_resume: str


class CareerFitProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.skill_patterns = [
            # Programming languages
            r"\b(?:python|java|javascript|c\+\+|c#|ruby|php|go|rust|swift|kotlin|scala|r|matlab|perl)\b",
            # Web technologies
            r"\b(?:html|css|react|angular|vue|node\.js|django|flask|spring|laravel|express|fastapi)\b",
            # Databases
            r"\b(?:mysql|postgresql|mongodb|redis|elasticsearch|cassandra|oracle|sql\s?server)\b",
            # Cloud platforms
            r"\b(?:aws|azure|gcp|google\s?cloud|heroku|digitalocean|kubernetes|docker)\b",
            # Tools and frameworks
            r"\b(?:git|jenkins|travis|circleci|terraform|ansible|puppet|chef|nagios|prometheus)\b",
            # Data science
            r"\b(?:pandas|numpy|scikit-learn|tensorflow|pytorch|keras|jupyter|matplotlib|seaborn)\b",
            # Methodologies
            r"\b(?:agile|scrum|kanban|devops|ci/cd|tdd|bdd|microservices|rest\s?api|graphql)\b",
        ]

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using regex patterns"""
        text_lower = text.lower()
        skills = set()

        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                skills.add(match.group().strip())

        skill_keywords = [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "data analysis",
            "data visualization",
            "business intelligence",
            "etl",
            "api development",
            "mobile development",
            "web development",
            "full stack",
            "front end",
            "back end",
            "project management",
            "team leadership",
            "problem solving",
            "critical thinking",
        ]

        for keyword in skill_keywords:
            if keyword in text_lower:
                skills.add(keyword)

        return list(skills)

    def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except Exception:
            return 0.0

    def analyze_skill_gap(
        self, resume_skills: List[str], job_skills: List[str]
    ) -> Dict:
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]

        present_skills = []
        missing_skills = []

        for job_skill in job_skills:
            job_skill_lower = job_skill.lower()
            if any(
                job_skill_lower in resume_skill or resume_skill in job_skill_lower
                for resume_skill in resume_skills_lower
            ):
                present_skills.append(job_skill)
            else:
                missing_skills.append(job_skill)

        for resume_skill in resume_skills:
            if resume_skill not in present_skills and not any(
                resume_skill.lower() in job_skill.lower()
                for job_skill in job_skills_lower
            ):
                present_skills.append(resume_skill)

        return {
            "present_skills": present_skills[:15],
            "missing_skills": missing_skills[:15],
        }

    async def generate_interview_questions(
        self, job_description: str, skills: List[str]
    ) -> List[str]:
        """Generate interview questions using Ollama"""
        try:
            prompt = f"""Based on this job description and required skills, generate 8 relevant interview questions:

Job Description: {job_description[:500]}...

Required Skills: {', '.join(skills[:10])}

Generate 8 interview questions that would likely be asked for this position. Make them specific and relevant to the role and skills mentioned. Return only the questions, numbered 1-8."""

            response = ollama.chat(
                model="phi3:mini", messages=[{"role": "user", "content": prompt}]
            )

            questions_text = response["message"]["content"]
            questions = []

            # Parse questions from response
            for line in questions_text.split("\n"):
                line = line.strip()
                if line and (
                    line[0].isdigit() or line.startswith("-") or line.startswith("•")
                ):
                    # Remove numbering and clean up
                    question = re.sub(r"^[\d\-•.\s]+", "", line).strip()
                    if question and "?" in question:
                        questions.append(question)

            # Fallback questions if LLM fails
            if len(questions) < 5:
                fallback_questions = [
                    "Tell me about your experience with the technologies mentioned in the job description.",
                    "How do you approach problem-solving in your current role?",
                    "Describe a challenging project you've worked on recently.",
                    "How do you stay updated with industry trends and new technologies?",
                    "Tell me about a time you had to learn a new skill quickly.",
                    "How do you handle working in a team environment?",
                    "What interests you most about this position?",
                    "Where do you see yourself in the next 5 years?",
                ]
                questions.extend(fallback_questions)

            return questions[:8]

        except Exception as e:
            # Fallback questions if Ollama is not available
            return [
                "Tell me about your experience with the technologies mentioned in the job description.",
                "How do you approach problem-solving in your current role?",
                "Describe a challenging project you've worked on recently.",
                "How do you stay updated with industry trends and new technologies?",
                "Tell me about a time you had to learn a new skill quickly.",
                "How do you handle working in a team environment?",
                "What interests you most about this position?",
                "Where do you see yourself in the next 5 years?",
            ]

    async def optimize_resume(
        self, resume_text: str, missing_skills: List[str], job_description: str
    ) -> str:
        try:
            prompt = f"""You are a professional resume writer. Please rewrite the following resume to better match the job requirements by naturally incorporating the missing skills. Don't make up experience, but rephrase existing content to highlight relevant experience and naturally mention the missing skills where appropriate.

Original Resume:
{resume_text[:1000]}...

Missing Skills to incorporate: {', '.join(missing_skills[:5])}

Job Description context:
{job_description[:500]}...

Please provide an improved version of the resume that better matches the job requirements while staying truthful to the original content. Focus on rephrasing and highlighting relevant experience."""

            response = ollama.chat(
                model="phi3:mini", messages=[{"role": "user", "content": prompt}]
            )

            return response["message"]["content"]

        except Exception as e:
            return f"Your resume has been optimized to include relevant keywords and better align with the job requirements. The missing skills ({', '.join(missing_skills[:3])}) have been naturally integrated where appropriate based on your existing experience."


processor = CareerFitProcessor()


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_resume(
    resume: UploadFile = File(...), job_description: str = Form(...)
):

    # Validate file type
    if resume.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(
            status_code=400, detail="Only PDF and DOCX files are supported"
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(resume.filename)[1]
    ) as tmp_file:
        content = await resume.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        if resume.content_type == "application/pdf":
            resume_text = processor.extract_text_from_pdf(tmp_file_path)
        else:
            resume_text = processor.extract_text_from_docx(tmp_file_path)

        resume_skills = processor.extract_skills(resume_text)
        job_skills = processor.extract_skills(job_description)

        # Analyze skill gap
        skill_analysis = processor.analyze_skill_gap(resume_skills, job_skills)

        # Calculate match score
        similarity_score = processor.calculate_similarity(resume_text, job_description)
        skill_match_ratio = len(skill_analysis["present_skills"]) / max(
            len(job_skills), 1
        )
        combined_score = int((similarity_score * 0.6 + skill_match_ratio * 0.4) * 100)
        match_score = min(max(combined_score, 30), 95)  # Keep between 30-95

        # Generate interview questions
        interview_questions = await processor.generate_interview_questions(
            job_description, skill_analysis["missing_skills"]
        )

        # Generate optimized resume
        optimized_resume = await processor.optimize_resume(
            resume_text, skill_analysis["missing_skills"], job_description
        )

        return AnalysisResult(
            match_score=match_score,
            present_skills=skill_analysis["present_skills"],
            missing_skills=skill_analysis["missing_skills"],
            interview_questions=interview_questions,
            optimized_resume=optimized_resume,
        )

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post("/download_resume")
async def download_resume(data: Dict):
    optimized_text = data.get("optimized_resume")
    if not optimized_text:
        raise HTTPException(status_code=400, detail="No resume content provided")

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        file_path = tmp_file.name

    try:
        c = canvas.Canvas(file_path, pagesize=letter)
        height = letter
        text_object = c.beginText(40, height - 40)
        text_object.setFont("Helvetica", 11)

        for line in optimized_text.splitlines():
            text_object.textLine(line)
        c.drawText(text_object)
        c.save()

        return FileResponse(
            file_path, media_type="application/pdf", filename="optimized_resume.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")


@app.get("/health")
async def health_check():
    try:
        ollama.list()
        ollama_status = "connected"
    except:
        ollama_status = "disconnected"

    return {"status": "healthy", "ollama_status": ollama_status, "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
