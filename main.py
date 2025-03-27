from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import base64
import io
from PIL import Image
import google.generativeai as genai

# Load environment variables and configure Gemini AI with the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Document Data Extractor API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gemini_response(prompt: str, image_content: dict) -> str:
    """
    Calls the Gemini AI model 'gemini-1.5-flash' to extract JSON data from the image.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, image_content])
    return response.text

def input_image_setup(file_bytes: bytes) -> dict:
    """
    Processes the uploaded image bytes: converts it to JPEG (handles alpha channels) and encodes it in base64.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Convert to RGB if the image is not already in that mode (removes alpha channel)
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()
        # Prepare the payload for Gemini AI
        image_data = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_bytes).decode()
        }
        return image_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    instructions: str = Form("")
):
    """
    Endpoint that accepts an image file and optional additional instructions.
    Returns JSON extracted from the document image.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    try:
      
        file_bytes = await file.read()
        image_content = input_image_setup(file_bytes)
        
    
        extraction_prompt = (
            "You are an expert document data extractor. Analyze the provided image of a document of any type "
            "(e.g., form, invoice, letter, etc.) and identify all relevant fields and information. "
            "Return the extracted data in a structured JSON format with descriptive keys, without any additional text. "
            "If certain typical fields are not present, simply omit them."
        )
        if instructions.strip():
            extraction_prompt += "\nAdditional instructions: " + instructions.strip()
        
        # Get the JSON response from Gemini AI
        extracted_data = get_gemini_response(extraction_prompt, image_content)
        return {"extracted_data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
