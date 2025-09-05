import { useState, useCallback } from "react";
import {
  Upload,
  FileText,
  Target,
  MessageSquare,
  Download,
  Loader2,
  CheckCircle,
} from "lucide-react";

const CareerFitAI = () => {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const handleResumeUpload = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      if (
        file.type === "application/pdf" ||
        file.type ===
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      ) {
        setResumeFile(file);
        setError("");
      } else {
        setError("Please upload a PDF or DOCX file");
      }
    }
  }, []);

  const handleJobDescriptionChange = useCallback((e) => {
    setJobDescription(e.target.value);
  }, []);

  const processFiles = async () => {
    if (!resumeFile || !jobDescription.trim()) {
      setError("Please upload a resume and enter a job description");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("resume", resumeFile);
      formData.append("job_description", jobDescription);

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to analyze resume");
      }

      const data = await response.json();

      setResults({
        matchScore: data.match_score,
        missingSkills: data.missing_skills,
        presentSkills: data.present_skills,
        interviewQuestions: data.interview_questions,
        optimizedResume: data.optimized_resume,
      });

      setStep(2);
    } catch (err) {
      if (err.message.includes("fetch")) {
        setError(
          "Cannot connect to server. Make sure the backend is running on http://localhost:8000"
        );
      } else {
        setError(err.message || "Failed to process files. Please try again.");
      }
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const downloadResume = async () => {
    try {
      const response = await fetch("http://localhost:8000/download_resume", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ optimized_resume: results.optimizedResume }),
      });

      if (!response.ok) throw new Error("Failed to download resume");

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "optimized_resume.pdf";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download error:", err);
    }
  };

  const resetApplication = () => {
    setStep(1);
    setResumeFile(null);
    setJobDescription("");
    setResults(null);
    setError("");
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            CareerFit AI
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered resume optimization and job matching
          </p>
        </div>

        {step === 1 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-2">
                Upload Your Documents
              </h2>
              <p className="text-gray-600">
                Upload your resume and paste the job description to get started.
              </p>
            </div>

            {/* Resume Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <FileText className="inline w-4 h-4 mr-2" />
                Resume Upload (PDF or DOCX)
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <div className="text-sm text-gray-600">
                  <label className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500">
                    <span>Upload a file</span>
                    <input
                      type="file"
                      className="sr-only"
                      accept=".pdf,.docx"
                      onChange={handleResumeUpload}
                    />
                  </label>
                  <p className="pl-1 inline">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">PDF or DOCX up to 10MB</p>
                {resumeFile && (
                  <div className="mt-2 flex items-center justify-center">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    <span className="text-sm text-green-600">
                      {resumeFile.name}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Job Description */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Target className="inline w-4 h-4 mr-2" />
                Job Description
              </label>
              <textarea
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows="8"
                placeholder="Paste the job description here..."
                value={jobDescription}
                onChange={handleJobDescriptionChange}
              />
            </div>

            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}

            <button
              onClick={processFiles}
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex flex-col items-center justify-center transition-colors"
            >
              {loading ? (
                <>
                  <div className="flex items-center">
                    <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" />
                    Processing...
                  </div>
                  <p className="text-xs text-gray-100 mt-2">
                    This may take a few minutes, please wait...
                  </p>
                </>
              ) : (
                "Analyze & Optimize"
              )}
            </button>
          </div>
        )}

        {step === 2 && results && (
          <div className="space-y-6">
            {/* Match Score */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                Match Score
              </h2>
              <div className="flex items-center">
                <div className="text-4xl font-bold text-blue-600 mr-4">
                  {results.matchScore}/100
                </div>
                <div className="flex-1">
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div
                      className="bg-blue-600 h-4 rounded-full transition-all duration-1000"
                      style={{ width: `${results.matchScore}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Your resume matches {results.matchScore}% of the job
                    requirements
                  </p>
                </div>
              </div>
            </div>

            {/* Skills Analysis */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-green-600">
                  Present Skills
                </h3>
                <div className="flex flex-wrap gap-2">
                  {results.presentSkills.map((skill, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-4 text-red-600">
                  Missing Skills
                </h3>
                <div className="flex flex-wrap gap-2">
                  {results.missingSkills.map((skill, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Optimized Resume */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                ATS-Optimized Resume
              </h3>
              <p className="text-gray-600 mb-4">{results.optimizedResume}</p>
              <button
                onClick={downloadResume}
                className="bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 flex items-center transition-colors"
              >
                <Download className="mr-2 h-4 w-4" />
                Download Optimized Resume
              </button>
            </div>

            {/* Interview Questions */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                <MessageSquare className="inline w-5 h-5 mr-2" />
                Likely Interview Questions
              </h3>
              <div className="space-y-3">
                {results.interviewQuestions.map((question, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded-lg">
                    <span className="font-medium text-blue-600">
                      Q{index + 1}:{" "}
                    </span>
                    {question}
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={resetApplication}
              className="w-full bg-gray-600 text-white py-3 px-4 rounded-lg hover:bg-gray-700 transition-colors"
            >
              Start New Analysis
            </button>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="text-center text-gray-500 text-sm mt-8">
        © {new Date().getFullYear()} Sarkhail. All rights reserved.
      </footer>
    </div>
  );
};

export default CareerFitAI;
