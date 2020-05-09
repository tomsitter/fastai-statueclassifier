from fastapi import FastAPI
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastai.vision import (
    ImageDataBunch,
    cnn_learner,
    open_image, 
    models,
    get_transforms,
    imagenet_stats
)
from io import BytesIO
from pathlib import Path
import aiohttp
import asyncio

app = FastAPI()

class Prediction(BaseModel):
    name = "StatueLearner"
    pred_ts: Optional[datetime] = None
    predictions: List[tuple] = []


path = '.'
classes = ['chinese', 'egyptian', 'greek'] 
data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learner = cnn_learner(data, models.resnet50)
learner.load('stage-1-resnet50-er19')

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/classify")
async def classify(url):
    bytes = await get_bytes(url)
    img = open_image(BytesIO(bytes))
    
    #losses = [9.9458e-01, 3.4059e-07, 5.4165e-03]
    _,_,losses = learner.predict(img)
    return Prediction(
        pred_ts = datetime.now(),
        predictions = sorted(
            zip(classes, map(float, losses)),
            key= lambda p: p[1],
            reverse=True
        ),
    )