# CareerFit AI

A smart assistant that helps job seekers **match their resumes with job descriptions**, identify missing skills, optimize resumes for ATS systems, and prepare for interviews.  

The backend is built with **Python + FastAPI** and powered by a local **LLM**. The frontend is built with **React + Vite + Tailwind CSS**.

---

## Core Features

- **Resume Upload**  
  Upload a resume (PDF/DOCX). Text is extracted using Python libraries such as `PyMuPDF` and `docx2txt`.

- **Job Description Upload/Paste**  
  Paste or upload a job description for analysis.

- **Skill Extraction & Matching**  
  Extracts keywords and skills from both resume and job description.

- **Skill Gap Analysis**  
  Finds missing skills that are in the job description but not in the resume.

- **ATS-Optimized Resume Rewrite**  
  Rewrites the resume by rephrasing and naturally inserting missing skills.

- **Interview Question Generation**  
  Generates 8–10 likely interview questions tailored to the job description.

- **Match Score**  
  Calculates semantic similarity and provides a score out of 100 for fit.

---

## Prerequisites

- **Python** (>= 3.10)  
- **Ollama** installed locally ([Download here](https://ollama.com/download))  
- A pulled model (choose one):

```bash
# Higher-quality model
ollama pull llama3.1:8b

# Smaller, faster model
ollama pull phi3:mini
```

## Backend Setup (FastAPI)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run the API
```bash
uvicorn main:app --reload --port 8000
```

## Frontend Setup (React + Vite + Tailwind)
```bash
cd frontend
```

## Install Dependencies
```bash
npm install
```

## Start the dev server
```bash
npm run dev
```

**Open the printed URL (typically http://127.0.0.1:5173)**
