from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title='Home Credit Prediction')


def load_model():
    with open('model_rf_v2.sav','rb') as file:
        return pickle.load(file)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/predict')
def get_prediction(request: list[dict]):
    data = pd.DataFrame(request)
    model = load_model()
    predict_prob = model.predict_proba(data)[:,1].tolist()
    predict_class = [1 if i >= 0.5 else 0 for i in predict_prob]
    return {'class':predict_class, 'prob':predict_prob}
