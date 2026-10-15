import numpy as np
import tensorflow as tf
import streamlit as st

st.title("📚 Previsão de Horas de Estudo - ENEM")
st.subheader("Exemplo Prático de Regressão com TensorFlow")

st.markdown("""
Esta aplicação utiliza uma **Rede Neural Simples (1 neurônio)** para aprender 
a relação entre as notas obtidas no ENEM e as horas de estudo dedicadas.
""")

# 1. Dicionário de dados simulado (Base de Conhecimento)
notas = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
horas_estudo = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

# 2. Função com cache para treinar a Rede Neural apenas uma vez
@st.cache_resource
def treinar_modelo():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    modelo.compile(optimizer='sgd', loss='mean_squared_error')
    modelo.fit(notas, horas_estudo, epochs=500, verbose=0)
    return modelo

with st.spinner("Treinando a rede neural com o TensorFlow..."):
    modelo = treinar_modelo()

st.success("Modelo treinado com sucesso!")

st.divider()

# 3. Entradas do Usuário via Streamlit
nota_desejada = st.slider(
    label="Escolha a nota desejada no ENEM:",
    min_value=5.0,
    max_value=10.0,
    value=10.0,
    step=0.5
)

# 4. Previsão da Rede Neural
previsao = modelo.predict(np.array([nota_desejada]))[0][0]

# 5. Exibição do Resultado
st.metric(
    label=f"Horas diárias de estudo recomendadas para nota {nota_desejada:.1f}",
    value=f"{previsao:.1f} hrs/dia"
)