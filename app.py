# Calculadora de precio de vivienda · Regresión lineal (California Housing)
# Ejecutar local:  streamlit run app.py   |  Publicar: Streamlit Community Cloud
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Precio de vivienda · Regresión lineal", page_icon="🏠")

@st.cache_resource
def entrenar():
    df = fetch_california_housing(as_frame=True).frame
    X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    p = m.predict(Xte)
    met = {"R²": r2_score(yte, p), "RMSE": np.sqrt(mean_squared_error(yte, p)), "MAE": mean_absolute_error(yte, p)}
    return m, X, y, met

modelo, X, y, met = entrenar()

st.title("🏠 Calculadora del precio de una vivienda")
st.write("Este modelo de **regresión lineal** estima el valor mediano de las viviendas de un distrito de California "
         "a partir de sus características. Mueve los controles de la barra lateral y mira cómo cambia la predicción "
         "y cuánto aporta cada variable.")

st.sidebar.header("Características del distrito")
v = {}
for c, etiqueta in [("MedInc", "Ingreso mediano (x $10.000)"), ("HouseAge", "Antigüedad de viviendas (años)"),
                    ("AveRooms", "Habitaciones promedio"), ("AveBedrms", "Dormitorios promedio"),
                    ("Population", "Población"), ("AveOccup", "Ocupantes promedio"),
                    ("Latitude", "Latitud"), ("Longitude", "Longitud")]:
    lo, hi, med = float(X[c].quantile(.01)), float(X[c].quantile(.99)), float(X[c].median())
    v[c] = st.sidebar.slider(etiqueta, lo, hi, med)

fila = pd.DataFrame([v])
pred = float(modelo.predict(fila)[0]) * 100_000
c1, c2 = st.columns(2)
c1.metric("Precio estimado", f"${pred:,.0f}")
c2.metric("Error típico del modelo (RMSE)", f"±${met['RMSE']*100_000:,.0f}")

# Gráfico 1: contribución de cada variable respecto al distrito "promedio"
contrib = (fila.iloc[0] - X.mean()) * modelo.coef_ * 100_000
fig, ax = plt.subplots(figsize=(6, 3.5))
contrib.sort_values().plot(kind="barh", ax=ax, color=["#c0392b" if x < 0 else "#2e86c1" for x in contrib.sort_values()])
ax.set_xlabel("Efecto sobre el precio vs. distrito promedio (USD)")
ax.set_title("¿Qué variables suben o bajan el precio?")
st.pyplot(fig)

# Gráfico 2: dónde cae la predicción respecto a los datos reales
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.hist(y * 100_000, bins=50, color="#d5dbdb")
ax2.axvline(pred, color="#c0392b", lw=2, label="Tu estimación")
ax2.set_xlabel("Precio (USD)"); ax2.legend(); ax2.set_title("Tu predicción frente a todos los distritos")
st.pyplot(fig2)

with st.expander("Calidad del modelo (datos de prueba)"):
    st.write(f"**R² = {met['R²']:.3f}**: explica ~{met['R²']*100:.0f}% de la variación del precio. "
             f"**MAE = ${met['MAE']*100_000:,.0f}** · **RMSE = ${met['RMSE']*100_000:,.0f}**.")
    st.caption("Datos del censo de 1990: las cifras sirven para aprender, no para tasar viviendas reales.")
