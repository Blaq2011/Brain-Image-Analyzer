
from fastapi import FastAPI, UploadFile, File
import os
import tempfile
from Test import predict_plane 


app = FastAPI()

# Optional: Handle root URL to avoid 404 error
@app.get("/")
async def root():
    return {"message": "Brain Image Analyzer API is running 🚀"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Create a unique temp file path using the system's temp directory
        suffix = os.path.splitext(file.filename)[-1]  # keep file extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Pass the path to your model
        prediction, confidence = predict_plane(tmp_path)

        return {"prediction": prediction, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as cleanup_err:
            print(f"Cleanup error: {cleanup_err}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render uses 10000 by default
    uvicorn.run("main:app", host="0.0.0.0", port=port)