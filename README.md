# mldeployment-cpe393

# Housing Price Prediction API
This project is a Flask-based REST API that serves a machine learning regression model trained to predict housing prices. 
The model is trained using scikit-learn on a housing dataset and deployed via a API with Docker support.

# install dependencies
pip install -r requirements.txt

# model export
Run train.py. (model.pkl will be saved in app folder)

# Go to the directory in terminal
cd "project folder directory"

# Build Docker image
docker build -t ml-model .

# Run Docker container
docker run -p 9000:9000 ml-model

# Test the API in POSTMAN
POST /predict
Content-Type: application/json

# Request Body 
{
  "features": [
    {
      "area": 3000,
      "bedrooms": 3,
      "bathrooms": 2,
      "stories": 2,
      "mainroad_yes": 1,
      "guestroom_yes": 0,
      "basement_yes": 1,
      "hotwaterheating_yes": 0,
      "airconditioning_yes": 1,
      "parking": 2,
      "furnishingstatus_semi-furnished": 1,
      "furnishingstatus_unfurnished": 0
    }
  ]
}

expected output

{"prediction": 4723051.234}

# Health Check
GET /health
expected output
{
  "status": "ok"
}



