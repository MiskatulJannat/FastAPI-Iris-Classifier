from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the trained model
with open('iris_model.pkl', 'rb') as f:
    clf = pickle.load(f)

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
async def predict_species(data: IrisData):
    try:
        # Convert the input data to a numpy array
        input_data = np.array([data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]).reshape(1, -1)
        
        # Predict the species
        prediction = clf.predict(input_data)
        
        # Map numeric labels to species names
        species_mapping = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }
        
        # Return the predicted species
        return {"species": species_mapping[prediction[0]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
