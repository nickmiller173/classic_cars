# 🚗 Classic Car Price Predictor
A full-stack machine learning application that predicts the auction price of classic cars. This project demonstrates an end-to-end data science pipeline: from scraping raw data and training a Random Forest model to deploying a serverless API on AWS and building a user-facing dashboard.

## 🏗️ Architecture Overview
The application is built using a Serverless Microservices architecture. This ensures the app is cost-effective (costs $0 when idle) and scalable.

Data Pipeline: Selenium Web Scraper ➡ Pandas Cleaning ➡ Scikit-Learn Training

Backend: AWS Lambda (Compute) ➡ AWS ECR (Container Storage) ➡ API Gateway (Public Interface)

Frontend: Streamlit (User Dashboard)

# 🔧 Pipeline Components
## 1. Data Collection (Scraping)
File: processing/scrape.py

Tool: Selenium (Python)

Target: Cars & Bids (Past Auctions)

How it works:

The script launches a headless Chrome browser to simulate a real user.

It scrolls through auction history pages to collect URLs.

It visits each auction page to extract key features: Year, Make, Model, Mileage, Engine, Drivetrain, Sold Price, etc.

Why Selenium? The target site uses dynamic JavaScript to load data, which standard requests (like BeautifulSoup) cannot handle.

## 2. Data Preprocessing & Feature Engineering
File: processing/preprocessing.ipynb

Tools: Pandas, NumPy

Key Challenges Solved:

Text Cleaning: Converted raw strings like "$45,000" to float 45000.0.

Standardization: Normalized varied inputs (e.g., mapping "Bianco Monocerus" and "Chalk" to "White").

Feature Extraction: Parsed engine strings (e.g., "4.0L Flat-6") into separate Displacement (4.0) and Cylinders (Flat-6) features.

## 3. Model Training
File: training/training.ipynb

Algorithm: Random Forest Regressor (Scikit-Learn)

Encoding Strategy:

Target Encoding: Used for the Model column (e.g., "911", "M3") to handle high cardinality. Each model is mapped to the average price of that car in the training set.

Label Encoding: Used for Make, Color, etc.

One-Hot Encoding: Used for low-cardinality features like Transmission and Body Style.

Artifact Generation: The training process saves three critical files needed for the app:

model.pkl: The trained brain.

encoding_artifacts.pkl: The translators (to convert "Red" ➡ 7).

app_options.json: A list of valid inputs for the frontend dropdowns.

## 4. Deployment (The Serverless Backend)
This is the core of the production system. We containerized the model to run on the cloud.

Docker (deployment/Dockerfile):

Since our model relies on specific versions of scikit-learn and pandas, we cannot use a standard server environment.

We built a Docker container based on the public.ecr.aws/lambda/python:3.9 image.

This container bundles the OS, Python, libraries, and our .pkl model files into a single, immutable unit.

AWS ECR (Elastic Container Registry):

We used sm-docker to build the image inside SageMaker and push it to a private repository (classic_car_predictor) in AWS.

Why? AWS Lambda cannot read files from a laptop; it pulls the secure image from ECR.

AWS Lambda (The Compute):

This is a serverless function that runs the app.py script.

Memory: Tuned to 512MB to handle the heavy Pandas/Scikit-learn load.

Timeout: Increased to 30s to allow for cold-start model loading.

AWS API Gateway:

Serves as the public "front door." It provides a secure HTTPS URL that receives JSON data from the website and passes it to Lambda.

## 5. Frontend (The Dashboard)
File: streamlit_app.py

Tool: Streamlit

How it works:

Loads app_options.json to populate dropdown menus (ensuring users can only select valid cars).

Takes user input, formats it into JSON, and POSTs it to the API Gateway URL.

Displays the returned price prediction instantly.

# 🚀 How to Run Locally
Prerequisites
Python 3.9+

Docker (optional, for building)

AWS CLI (configured)

## 1. Run the Frontend
You can run the dashboard on your laptop without touching AWS (it will still call the live API).

Bash
1. Clone the repo
git clone https://github.com/nickmiller173/classic_cars.git
cd classic_cars

2. Install dependencies
pip install streamlit requests

3. Run the app
streamlit run streamlit_app.py
## 2. Update the Model (For Developers)
If you retrain the model and want to update the live API:

Train: Run training.ipynb. Move the new .pkl files to deployment/.

Build: From the deployment/ folder, run:

Bash
sm-docker build . --repository classic_car_predictor:latest
Deploy: Go to AWS Lambda Console ➡ Images ➡ Deploy New Image.

# 📂 Project Structure

classic_cars/
├── data/                       # Raw and processed CSV files
├── processing/                 
│   ├── scrape.py               # Selenium scraper
│   └── preprocessing.ipynb     # Data cleaning pipeline
├── training/
│   ├── training.ipynb          # Model training & evaluation
│   └── app_options.json        # Allowed values for dropdowns
├── deployment/                 # Backend Code
│   ├── app.py                  # Lambda handler (entry point)
│   ├── utils.py                # Shared cleaning functions
│   ├── Dockerfile              # Container definition
│   └── *.pkl                   # Serialized model artifacts
└── streamlit_app.py            # Frontend User Interface