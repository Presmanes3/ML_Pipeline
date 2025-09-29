import streamlit as st
from utils.mlflow_utils import load_models
from utils.data_utils import load_data, get_input_features
from utils.plotting import plot_predictions, plot_price_distribution

st.title("🔮 Playground: Predict House Prices")

# Cargar modelos y dataset
df = load_data()
models = load_models()

# Inputs del usuario
input_data, sqft, dist = get_input_features(df)

# Calcular predicciones
results = {}
for model_label, model in models.items():
    pred_log = model.predict(input_data)[0]
    price = float(np.exp(pred_log))
    results[model_label] = {"price": price}

# Mostrar resultados
st.write("✅ Predictions:", results)
plot_predictions(results)
plot_price_distribution(df, results)
