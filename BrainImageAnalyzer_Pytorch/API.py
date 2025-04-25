from fastapi import FastAPI, UploadFile, File
import os
import tempfile
from .Test import predict_plane 
#import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://brain-image-analyzer.onrender.com"],  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # or be specific: ["POST"]
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Create a unique temp file path using the system's temp directory
        suffix = os.path.splitext(file.filename)[-1]  # keep file extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name  # ✅ Correct assignment

        # Pass the path to your model
        prediction, confidence = predict_plane(tmp_path)

        return {"prediction": prediction, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up the temp file
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as cleanup_err:
            print(f"Cleanup error: {cleanup_err}")



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000)) 
    uvicorn.run("BrainImageAnalyzer_Pytorch.API:app", host="0.0.0.0", port=port)


# if __name__ == "__main__":
#     uvicorn.run("API:app", host="127.0.0.1", port=8080, reload=True)

# uvicorn BrainImageAnalyzer_Pytorch.API:app --host 127.0.0.1 --port 8080 