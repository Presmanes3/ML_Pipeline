import streamlit as st
import os, sys

from dotenv import load_dotenv

load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

sys.path.append(os.getcwd())

# Set the page configuration with a title and wide layout
st.set_page_config(page_title="House Price App", layout="wide")

# Display the main title of the app
st.title("🏠 House Price Prediction Platform")

# Add a markdown description for the app
st.markdown("""
    Welcome to the New York House Price Prediction Platform. This application is designed to help you explore and analyze data related to house prices, as well as to test prediction models.
    
    This application has been developed by [Javier Presmanes](https://www.linkedin.com/in/javierpresmanescardama/) as part of a machine learning project. You can find the source code on [GitHub](https://github.com/Presmanes3/ML_Pipeline).
    
    The data used in this project comes from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market) and has been cleaned and preprocessed for analysis.
    
    This application is divided into 3 main sections:
    1. **Problem Analysis (EDA)**: explore the data and its distributions.
    2. **Playground**: test the model with different inputs.
    3. **Business Insights**: gain insights from the data and model.
    
    Enjoy exploring and analyzing the data!
    """)
