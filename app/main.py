import datetime, os
from typing import Annotated
from fastapi import FastAPI
from pydantic import BaseModel
from WhisperModel import WhisperModel
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="Whisper Large", description="This is a Whisper Large model for speech recognition")
queue = {}
model = WhisperModel()

class Predict_Vars(BaseModel):
  remove_after: bool = True

@app.get("/")
def health_check():
    return {"health_check": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/help")
def help():
    return {"GET /": "Health check", 
            "GET /help": "Returns this help message", 
            "GET /info": "Returns information about the model and the device it is running on.",
            "GET /predict/{user_id}": "Predicts the text from the audio files in the queue of the user_id. Optionally removes the audio files after prediction.", 
            "GET /queue/{user_id}": "Returns the list of audio files in the queue of the user_id. Optionally removes the audio files after prediction with the remove_after query parameter.", 
            "POST /queue/{user_id}": "Adds audio file to the queue of the user_id. The audio file should be sent in the request body binary format. Returns the list of audio files in the queue of the user_id."}

@app.get("/info")
def get_info():
    return {"model": model.model_id, "device": model.device, "currently running": model.currently_running()}

@app.post("/queue/{user_id}")
def add_queue(user_id: str, audio_file: UploadFile):
    audio_file_path = os.path.join("data", audio_file.filename)

    print(f"Saving audio file to {audio_file_path}")
    with open(audio_file_path, "wb") as f: 
        f.write(audio_file.file.read())

    if user_id not in queue:
        queue[user_id] = set()
    
    queue[user_id].add(audio_file.filename)

    return get_queue(user_id)

@app.get("/queue/{user_id}")
def get_queue(user_id: str):
    return queue.get(user_id, [])

@app.get("/clear_queue/{user_id}")
def clear_queue(user_id: str):
    if user_id in queue:
        while len(queue[user_id]) > 0:
            audio_file_path = os.path.join("data", queue[user_id].pop())
            os.remove(audio_file_path)
    return get_queue(user_id)

@app.get("/predict/{user_id}")
def predict(user_id: str, remove: bool = True):
    if len(queue.get(user_id, [])) == 0:
        return {"error": "No audio files in the queue."}
    if model.currently_running():
        return {"error": "Model is currently running, please try again later."} 
    
    result = {}
    for audio_file in queue.get(user_id, []):
        audio_file_path = os.path.join("data", audio_file)

        text = model.get_prediction(audio_file_path=audio_file_path)
        result[audio_file_path] = text

    if remove:
        print("Removing audio files...")
        while len(queue.get(user_id, [])) > 0:
            audio_file_path = os.path.join("data", queue[user_id].pop())
            os.remove(audio_file_path)
    
    return result
