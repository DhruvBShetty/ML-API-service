from fastapi import FastAPI
import pickle
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):

    await load_models()
    yield

async def load_models():
    global com, model, sent_tf, sent
    com=pickle.load(open(os.getenv("COM_MODEL_PATH", 'text_tf.pkl'),'rb'))
    model = pickle.load(open(os.getenv("PROFANITY_MODEL_PATH", 'profanity.pkl'),'rb'))
    sent_tf=pickle.load(open(os.getenv("SENTIMENT_MODEL_PATH", 'text_sent_tf.pkl'),'rb'))
    sent=pickle.load(open(os.getenv("REVIEWER_MODEL_PATH", 'reviewer.pkl'),'rb'))

app = FastAPI(lifespan=lifespan)


origins = [
    "http://localhost:3000",
    "https://socialwordcloud.live"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():

    return {"message": "Hello World"}

@app.post("/")
async def review(text):
    try:
        tf_com=com.transform([text])
        ans=model.predict(tf_com)
        ans2=model.predict_proba(tf_com)

        sent_tx=sent_tf.transform([text])
        sentiment=sent.predict_proba(sent_tx)
        
        if(sentiment[0][0]<0.30):
            print(sentiment[0][0])
            sentiment="1"
        else:
            sentiment="0"
        
        
        if(ans2[0][0]>0.15):
            ans="Safe content"
        else:
            ans="Not appropriate :("

        return {"message": ans,"sentiment":sentiment}
    
    except Exception as e:
        return {"error": str(e)}

