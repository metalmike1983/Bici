
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import time

st.set_page_config(page_title="Contatore Biciclette", page_icon="🚲")
st.title("🚲 Biciclette Vendute in Tempo Reale")

# Data di partenza fissa
start_time = datetime(2024, 1, 1, 0, 0, 0)
now = datetime.utcnow()
minutes_passed = int((now - start_time).total_seconds() // 180)
vendite_reali = minutes_passed

# Session state per vendite simulate
if 'vendite_simulate' not in st.session_state:
    st.session_state.vendite_simulate = 0

# Bottone per simulare una nuova vendita
if st.button("🛒 Simula nuova vendita"):
    st.session_state.vendite_simulate += 1

# Totale vendite
totale_vendite = vendite_reali + st.session_state.vendite_simulate
st.metric("Biciclette vendute", f"{totale_vendite:,}".replace(",", "."))

# Grafico delle vendite simulate nel tempo (esempio fittizio)
st.subheader("📈 Andamento stimato delle vendite")
# Creiamo una serie temporale fittizia
times = [now - timedelta(days=i) for i in range(30)][::-1]
vendite_giornaliere = [int((i * 24 * 60) / 180 + st.session_state.vendite_simulate // 30) for i in range(30)]
df = pd.DataFrame({'Data': times, 'Biciclette vendute': vendite_giornaliere})
st.line_chart(df.set_index('Data'))

st.caption("⚙️ Una bicicletta viene venduta ogni 3 minuti, a partire dal 1° gennaio 2024. Simula vendite cliccando sul bottone.")
