import streamlit as st
import plotly.express as px
from utils import PlotManager, ConfigManager
import pandas as pd

# Config
data = ConfigManager.get_comment_data()

with st.sidebar:
    brand = st.selectbox("Choose the first brand", st.session_state['brands'] + ['All'], index = 13)
    
df_wordcloud = PlotManager.worcdloud_generate(data, brand)[0]
st.subheader("Wordcloud counts")
l, r = st.columns((0.2, 0.8))
with l:
    st.write(df_wordcloud)
with r:
    fig = px.bar(df_wordcloud, 'word', 'count')
    st.plotly_chart(fig)



st.subheader("Introduction")


st.subheader("Motivation & Problem Statement")
st.write('''
1. We want to verify the correlation between the rating and consumers' actual opinions about the 

         ''')

st.subheader("Data Source")


st.subheader("Methodology")



st.subheader("Results")



st.write("DESIGNING")
# *** brand average scores
# shops = ConfigManager.get_shop_data()
# brands = ConfigManager.get_brand_data(shops)
# fig = px.bar(brands, 'brand', '服務態度_score', hover_data = ['shop count'])
# fig.update_traces(marker = dict(color = '#E6DCB9'))
# st.plotly_chart(fig)