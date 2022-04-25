from fastapi import FastAPI, UploadFile, File
from app.prediction import read_imagefile, pre_processing, predict, prediction
import timeit as time

app = FastAPI()

@app.get("/") 
def home(): 
    return "/docs"

@app.post("/api/predict")
async def predict_basemodel_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg")
    if not extension:
        return "Image must be jpg format!"

    start = time.timeit()

    image = read_imagefile(await file.read())
    image_tensor = pre_processing(image)
    prediction = predict(image_tensor, start)

    return prediction

@app.post("/api/prediction")
async def get_prediction(file: bytes = File(...)):

    output = prediction(file)

    return output