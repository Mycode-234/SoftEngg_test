from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")  # Download from Firebase
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
model = tf.keras.models.load_model("model.h5")
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    label = class_names[np.argmax(predictions)]

    # Save to Firebase
    doc_ref = db.collection("predictions").add({"label": label})

    return {"prediction": label}
