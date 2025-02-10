"""
The module is a micro-service that can be integrated to provide Profanity and Sentiment
classification on movie reviews
"""
import os
import pickle
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.exceptions import NotFittedError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.ERROR, filename="errors.log")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Triggers loading of models and transformers on startup
    """
    await load_models()
    yield


async def load_models():
    """Loading models and transformers"""
    global com, model, sent_tf, sent

    with open(os.environ.get("COM_MODEL_PATH"), "rb") as f:
        com = pickle.load(f)

    with open(os.environ.get("PROFANITY_MODEL_PATH"), "rb") as f:
        model = pickle.load(f)

    with open(os.environ.get("SENTIMENT_MODEL_PATH"), "rb") as f:
        sent_tf = pickle.load(f)

    with open(os.environ.get("REVIEWER_MODEL_PATH"), "rb") as f:
        sent = pickle.load(f)


app = FastAPI(lifespan=lifespan)


origins = ["http://localhost:3000", "https://socialwordcloud.live"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    To check if backend is up and running.
    """
    return {"message": "ProfanitySentiment classifier is ready"}


@app.post("/")
async def review(text):
    """
    Endpoint receives text, transforms the text and receives the prediction from the model
    """
    try:
        tf_com = com.transform([text])
        ans = model.predict(tf_com)
        ans2 = model.predict_proba(tf_com)

        sent_tx = sent_tf.transform([text])
        sentiment = sent.predict_proba(sent_tx)

        if sentiment[0][0] < 0.30:
            sentiment = "1"
        else:
            sentiment = "0"

        if ans2[0][0] > 0.15:
            ans = "Safe content"
        else:
            ans = "Not appropriate :("

        return {"message": ans, "sentiment": sentiment}

    except NotFittedError:
        print("Error: Model is not fitted. Please train it first.")

    except ValueError as e:
        print(f"ValueError: {e}")  # Handles shape mismatches, wrong data types

    except MemoryError:
        print("MemoryError: Input is too large. Consider batch processing.")

    except AttributeError as e:
        print(f"AttributeError: {e}")  # If calling `.predict()` on NoneType

    except TypeError as e:
        print(f"TypeError: {e}")  # If input types are incompatible

    except RuntimeError as e:
        print(f"RuntimeError: {e}")  # Internal sklearn error
