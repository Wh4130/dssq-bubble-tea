import streamlit as st
import plotly.express as px
from utils import PlotManager, ConfigManager

st.write("DESIGNING")
# *** brand average scores
shops = ConfigManager.get_shop_data()
brands = ConfigManager.get_brand_data(shops)
fig = px.bar(brands, 'brand', '服務態度_score', hover_data = ['shop count'])
fig.update_traces(marker = dict(color = '#E6DCB9'))
st.plotly_chart(fig)