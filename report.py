import streamlit as st
import plotly.express as px
from utils import PlotManager, ConfigManager
import pandas as pd
import statsmodels.api as sm

# *** Config
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

# todo wordcloud for huihui 
st.subheader("Wordcloud counts")
brand = st.selectbox("Choose the brand", st.session_state['brands'] + ['All'], index = 13)
df_wordcloud = PlotManager.worcdloud_generate(comments, brand)[0]
l, r = st.columns((0.2, 0.8))
with l:
    st.write(df_wordcloud)
with r:
    fig = px.bar(df_wordcloud, 'word', 'count')
    st.plotly_chart(fig)


# *** Introduction
st.subheader("Introduction")

# *** Motivation & Problem Statement
st.subheader("Motivation & Problem Statement")
st.write('''
1. We want to verify the correlation between the rating and consumers' actual opinions about the 

         ''')

# *** Data Source
st.subheader("Data Source")

# *** Methodology
st.subheader("Methodology")

# **************************************** Results ****************************************
st.subheader("Results")

# *** average rating & average sentiment
st.markdown("<h4> Average rating & Average sentiment</h4>", unsafe_allow_html=True)

st.write('''
In order to verify the hypothesis, we conduct ordinary least square regression for the rescaled average sentiment scores on the rating score across shop. The sentiment scores is calculated for all comments then grouped by "shop id" to calculate the average as the proxy of shop level scores; the rating score is directly scraped with **selenium** package from google review.''')

st.latex(r'AverageSentiment = \beta_0 + \beta_1 * AverageRating')
st.latex(r'H_0: \beta_1 = 0; H_1: \beta_1 \neq 0')

st.write('''Our null hypothesis posits that Google Map ratings do not accurately reflect people's sentiment toward bubble tea shops, while the alternative hypothesis asserts that Google Map ratings do represent comment sentiment. To test this, we perform an OLS regression with shop-level sentiment as the dependent variable and shop rating as the independent variable.       
''')

reg1_plot, reg1_table = st.columns((0.55, 0.45))

with reg1_plot:
    data_reg1 = ConfigManager.remove_outliers(shops, "avg_sentiment", 1.5)
    data_reg1 = ConfigManager.remove_outliers(data_reg1, "average_rating", 1.5)
    fig_reg1 = PlotManager.rating_regplot(data_reg1, 'All', ' ')
    event = st.plotly_chart(fig_reg1, on_select = 'rerun', key = 'reg1')

with reg1_table:
    # m1_X = sm.add_constant(data_reg1['average_rating'])
    # m1_Y = data_reg1['avg_sentiment']
    # model_1 = sm.OLS(m1_Y, m1_X).fit()
    # st.code(model_1.summary())
    st.code('''
OLS Regression Results 
=========================================          
Dep. Variable:              avg_sentiment  
Model:                                OLS   
Method:                     Least Squares   
R-squared:                          0.120
F-statistic:                        55.28
Prob (F-statistic):              6.29e-13
No. Observations:                     407   
                                    
=========================================
                    coef            P>|t|      
-----------------------------------------
const              2.4131           0.000      
average_rating     0.1500           0.000       
=========================================
''') 
    
st.write('''The regression analysis indicates a significant and positive correlation between shop-level sentiment scores and shop-level ratings, with a p-value approaching 0. The model's R-squared value is 0.12, suggesting that shop ratings can account for up to 12% of the variation in sentiment scores. This is notably substantial, considering the model includes only a single independent variable.     
''')




# *** distance & comment / rating counts
st.markdown("<h4>Distance to the nearest MRT exit & Comment Counts</h4>", unsafe_allow_html=True)

reg2_l, reg2_r = st.columns((0.55, 0.45))

with reg2_l:
    data_reg2 = ConfigManager.remove_outliers(shops, 'total_comment_or_rating_num', 1.5)
    data_reg2 = ConfigManager.remove_outliers(data_reg2, 'distance_to_mrt', 1.5)
    fig_reg2 = px.scatter(
            data_reg2,
            'distance_to_mrt',
            'total_comment_or_rating_num',
            trendline = 'ols',
            title = " ",
            hover_name = 'name'
        )
    
    fig_reg2.update_traces(marker = dict(size = 8, color = 'skyblue', opacity = 0.8))
    fig_reg2.update_layout(title_x = 0.45)
    event = st.plotly_chart(fig_reg2, on_select = 'rerun', key = 'reg2')

with reg2_r:
    # m2_X = sm.add_constant(data_reg2['distance_to_mrt'])
    # m2_Y = data_reg2['total_comment_or_rating_num']
    # model_2 = sm.OLS(m2_Y, m2_X).fit()
    # st.code(model_2.summary())
    st.code('''
OLS Regression Results                                
==========================================
Dep. Variable: # numtotal comments/scores   
Model:                                OLS   
Method:                     Least Squares   
R-squared:                          0.023
F-statistic:                        8.741
Prob (F-statistic):               0.00331
No. Observations:                     366
               
=========================================
                      coef          P>|t|      
-----------------------------------------
const             260.2293          0.000     
distance_to_mrt    -0.1118          0.003     
=========================================
''')
    
# *** distance & comment / average rating 
st.markdown("<h4>Distance to the nearest MRT exit & Average Rating</h4>", unsafe_allow_html=True)

reg3_l, reg3_r = st.columns((0.55, 0.45))

with reg3_l:
    data_reg3 = ConfigManager.remove_outliers(shops, 'distance_to_mrt', 1.5)
    data_reg3 = ConfigManager.remove_outliers(data_reg3, 'average_rating', 1.5)
    fig_reg3 = px.scatter(
            data_reg3,
            'distance_to_mrt',
            'average_rating',
            trendline = 'ols',
            title = " ",
            hover_name = 'name'
        )
    
    fig_reg3.update_traces(marker = dict(size = 8, color = 'orange', opacity = 0.8))
    fig_reg3.update_layout(title_x = 0.45)
    event = st.plotly_chart(fig_reg3, on_select = 'rerun', key = 'reg3')

with reg3_r:
    # m3_X = sm.add_constant(data_reg3['distance_to_mrt'])
    # m3_Y = data_reg3['average_rating']
    # model_3 = sm.OLS(m3_Y, m3_X).fit()
    # st.code(model_3.summary())
    st.code('''
OLS Regression Results                   
=========================================
Dep. Variable:             average_rating
Model:                                OLS
Method:                     Least Squares
R-squared:                          0.030     
F-statistic:                        12.19
Prob (F-statistic):              0.000536
No. Observations:                     397
                                       
=========================================
                      coef          P>|t|
-----------------------------------------
const               4.0017          0.000
distance_to_mrt     0.0004          0.001
=========================================
            ''')
#     mode3_1 = 


st.write("DESIGNING")
# *** brand average scores
# shops = ConfigManager.get_shop_data()
# brands = ConfigManager.get_brand_data(shops)
# fig = px.bar(brands, 'brand', '服務態度_score', hover_data = ['shop count'])
# fig.update_traces(marker = dict(color = '#E6DCB9'))
# st.plotly_chart(fig)