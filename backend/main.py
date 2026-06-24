import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from utils.extraction import extract_from_pdf, extract_from_image
from utils.scoring import calculate_ats_score

app = FastAPI(title="ATS CV Score Checker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    cv_text: str
    job_description_text: str

@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF or Image file, extracts text and returns it.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        contents = await file.read()
        file_ext = file.filename.split(".")[-1].lower()
        
        extracted_text = ""
        if file_ext == "pdf":
            extracted_text = extract_from_pdf(io.BytesIO(contents))
        elif file_ext in ["png", "jpg", "jpeg"]:
            extracted_text = extract_from_image(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
            
        return {"extracted_text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/analyze/")
async def analyze_cv(request: AnalyzeRequest):
    """
    Analyzes CV text against the Job Description and returns an ATS score.
    """
    if not request.cv_text or not request.job_description_text:
        raise HTTPException(status_code=400, detail="CV text and Job Description text are required")
        
    try:
        results = calculate_ats_score(request.cv_text, request.job_description_text)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing CV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
