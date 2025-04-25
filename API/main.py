from fastapi import FastAPI, UploadFile, File
import os
import tempfile
from Test import predict_plane 
#import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


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

