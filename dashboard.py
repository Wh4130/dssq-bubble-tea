import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import folium
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
import plotly.express as px


from utils import (
    ConfigManager,
    PlotManager
)

# *** HTML & CSS config

st.markdown(
        """
        <style>
            div[data-testid="stButtonGroup"]
            {
                text-align: center;
                align-items: center;
                display: flex;
                justify-content: center;
            }
        """,unsafe_allow_html=True
    )
st.markdown(
        """
        <style>
            div[data-testid="stMarkdownContainer"]
            {
                text-align: center;
                align-items: center;
                display: flex;
                justify-content: center;
            } 
        """,unsafe_allow_html=True
    )
st.markdown(
        """
        <style>
            div[data-testid="stMarkdown"]
            {
                text-align: center;
                align-items: center;
                display: flex;
                justify-content: center;
            } 
        """,unsafe_allow_html=True
    )
st.markdown(
        """
        <style>
            div[data-testid="stButton"]
            {
                text-align: center;
                align-items: center;
                display: flex;
                justify-content: center;
                width: 100%;
            }
        """,unsafe_allow_html=True
    )

# *** get data
with st.spinner("Loading data..."):
    shops = ConfigManager.get_shop_data()
    comments = ConfigManager.get_comment_data()
    brands = ConfigManager.get_brand_data(shops)
    metric_explanation = ConfigManager.metric_explanation()
    mrt_stations = ConfigManager.get_mrt_data()

# *** get utils and set session states
st.session_state['brand_color_mapping'] = ConfigManager.brand_color_mapping()
st.session_state['brands'] = ConfigManager.significant_brands()

    
# *** User input
# input_left, input_right = st.columns((1/2, 1/2))
# with input_left:
#     brand1 = st.selectbox("Choose one brand", st.session_state['brands'] + ['All'], index = 0)
# with input_right:
#     brand2 = st.selectbox("Choose one brand", st.session_state['brands'] + ['All'], index = 2)

# *** sidebar
with st.sidebar:
    brand1 = st.selectbox("Choose the first brand", st.session_state['brands'] + ['All'], index = 13)
    brand2 = st.selectbox("Choose the second brand", st.session_state['brands'] + ['All'], index = 16)

    b1, b2 = st.columns(2)
    with b1:
        if st.button('Reload', type = 'secondary'):
            st.cache_data.clear()
            st.rerun()
    with b2:
        if st.button('Rerun', type = 'primary'):
            st.rerun()
    
# *** Hint for users
st.info("You can expand the sidebar to select two brands for more detailed comparison", icon = '⚠️')

# st.map(data = mrt_stations, latitude = 'latitude', longitude = 'longtitude')

        


# *** brand average scores
with st.container(border = True):
    st.markdown("<h3 style='text-align: center; '>Overall Comparison</h3>", unsafe_allow_html=True)
    st.markdown("<span style='text-align: center; '>Select one metric for comparison</span>", unsafe_allow_html=True)
    metric = st.segmented_control(" ", ['average_rating', '品項_score', '口味_score', '服務態度_score', 'avg_sentiment', 'shop count'], format_func = lambda x: x.replace("_", " ").replace("avg", "average"), default = 'average_rating')
    if metric is not None:
        st.caption(metric_explanation[metric])
        st.plotly_chart(PlotManager.brands_barplot(brands, metric))
    else:
        st.warning("Please select one metric", icon = '⚠️')

# *** wordclouds 
with st.container(border = True):
    st.markdown("<h3 style='text-align: center; '>Comparison for two brands</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; '>Wordclouds</h4>", unsafe_allow_html=True)
    st.markdown("<span style='text-align: center; '>Select multiple dimensions of interest from below</span>", unsafe_allow_html=True)
    dims = st.segmented_control(" ", ['品項', '口味', '服務態度'], format_func = lambda x: f"{x} ({metric_explanation[x]})", selection_mode = 'multi', default = '品項')
    comments_filtered_by_dims = comments.loc[comments['category'].isin(dims), :].dropna()
    
    if dims != []:
        
        
        with st.container():
            box2_left, box2_right = st.columns(2)
            with box2_left:
                st.markdown(f"<h6 style='text-align: center; '>{brand1}</h6>", unsafe_allow_html=True)
                with st.container(height = 100):
                    st.caption(PlotManager.random_pick_comment(comments_filtered_by_dims, brand1))
                wc_left = PlotManager.worcdloud_generate(comments_filtered_by_dims, brand1)[1]
                st.pyplot(wc_left)

            with box2_right:
                st.markdown(f"<h6 style='text-align: center; '>{brand2}</h6>", unsafe_allow_html=True)
                with st.container(height = 100):
                    st.caption(PlotManager.random_pick_comment(comments_filtered_by_dims, brand2))    
                wc_right = PlotManager.worcdloud_generate(comments_filtered_by_dims, brand2)[1]
                st.pyplot(wc_right)

        box3_left, box3_right = st.columns(2)

    else:
        pass

    st.divider()

# *** scatter plots
    st.markdown("<h3 style='text-align: center; '>Averaging Rating & Scaled Sentiment-Based Rating</h3>", unsafe_allow_html=True)
    box1_left, box1_right = st.columns((1/2, 1/2))
    with box1_left:
        fig_box1_left = PlotManager.rating_regplot(shops, brand1, brand1)
        event = st.plotly_chart(fig_box1_left, on_select = 'rerun', key = 'left')
    with box1_right:
        fig_box1_right = PlotManager.rating_regplot(shops, brand2, brand2)
        event = st.plotly_chart(fig_box1_right, on_select = 'rerun', key = 'right')

# *** geographical distribution
    st.markdown("<h3 style='text-align: center; '>Geographical Distribution</h3>", unsafe_allow_html=True)
    st.caption(f"size: average sentiment score | blue dots: {brand1} | red dots:  {brand2}")
    m = PlotManager.init_map(mrt_stations)
    m = PlotManager.add_shops_to_map(m, shops, brand1, 'blue')
    m = PlotManager.add_shops_to_map(m, shops, brand2, 'red')
    st_folium(m, height = 500, width = 1300)
    


# *** raw data tables
with st.expander('Raw Data'):
    with st.container(key = 'dfs'):
        st.markdown("<h3 style='text-align: center; '>Shops</h3>", unsafe_allow_html=True)
        box3_left, box3_right = st.columns((1/2, 1/2))
        with box3_left:
            st.markdown(f"<h6 style='text-align: center; '>{brand1}</h6>", unsafe_allow_html=True)
            if brand1 == 'All':
                st.write(shops)
            else:
                st.write(shops[shops['brand'] == brand1])
        with box3_right:
            st.markdown(f"<h6 style='text-align: center; '>{brand2}</h6>", unsafe_allow_html=True)
            if brand2 == 'All':
                st.write(shops)
            else:
                st.write(shops[shops['brand'] == brand2])
        st.markdown("<h3 style='text-align: center; '>Comments</h3>", unsafe_allow_html=True)
        box4_left, box4_right = st.columns(2)
        with box4_left:
            st.markdown(f"<h6 style='text-align: center; '>{brand1}</h6>", unsafe_allow_html=True)
            if brand1 == 'All':
                st.write(comments)
            else:
                st.write(comments[comments['brand'] == brand1])
        with box4_right:
            st.markdown(f"<h6 style='text-align: center; '>{brand2}</h6>", unsafe_allow_html=True)
            if brand2 == 'All':
                st.write(comments)
            else:
                st.write(comments[comments['brand'] == brand2])


        st.markdown("<h3 style='text-align: center; '>Brands</h3>", unsafe_allow_html=True)
        st.write(brands)











