import streamlit as st

# Set the page configuration with a title and wide layout
st.set_page_config(page_title="House Price App", layout="wide")

# Display the main title of the app
st.title("🏠 House Price Prediction Platform")

# Add a markdown description for the app
st.markdown("""
    Bienvenido a la plataforma de predicción de precios de viviendas en Nueva York. Esta aplicación está diseñada para ayudarte a explorar y analizar datos relacionados con los precios de las viviendas, así como para probar modelos de predicción.
    
    Esta aplicacion ha sido desarrollada por [Javier Presmanes](https://www.linkedin.com/in/javierpresmanescardama/) como parte de un proyecto de aprendizaje automático. Puedes encontrar el código fuente en [GitHub](https://github.com/Presmanes3/ML_Pipeline).
    
    Los datos utilizados en este proyecto provienen de [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market) y han sido limpiados y preprocesados para su análisis.
    
    Esta aplicacion esta dividida en 4 secciones principales:
    1. **Problem Analysis (EDA)**: explore the data and its distributions.
    2. **Pipeline & Model**: explicacion acerca de la pipeline de preprocesamiento y el modelo utilizado.
    3. **Playground**: test the model with different inputs.
    4. **Business Insights**: gain insights from the data and model.
    
    ¡Disfruta explorando y analizando los datos!
    """)
