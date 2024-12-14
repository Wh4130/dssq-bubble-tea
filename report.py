import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import PlotManager, ConfigManager
import pandas as pd
import statsmodels.api as sm
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# *** Config
plt.rcParams['font.family'] = 'Osaka'

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


with st.sidebar:
    st.markdown("""
- Index
    - [Motivation](#Motivation)
    - [Hypotheses](#Hypotheses)
    - [Data Source](#Data_Source)
    - [Methodology](#Methodology)
    - [Results](#Results)
- <a href="https://github.com/Wh4130/dssq-bubble-tea">Github Repo</a>
""", unsafe_allow_html = True)

# *** Motivation
st.header("Motivation", anchor = "Motivation")
st.write('''Bias in Google Maps ratings has been an ongoing issue in Taiwan, potentially leading viewers to choose overrated restaurants and resulting in negative experiences. As fans of bubble tea, we feel that similar rating bias may also affect Google Maps reviews of bubble tea shops. This has motivated us to analyze and re-rate bubble tea shops to provide more accurate recommendations. This project's main purpose is **to verify whether bubble tea shops' rating actually reflects people's view which we measures by the average sentiment score of comments per shop. **''')
st.divider()

# *** Hypothesis
st.header("Hypotheses", anchor = "Hypotheses")
st.write('''
1. Our **main hypothesis** focuses on the existance of the correlation between actual rating and average sentiment score: 
         
    The null hypothesis posits that Google Map ratings do not accurately reflect people's sentiment toward bubble tea shops, while the alternative hypothesis asserts that Google Map ratings do represent comment sentiment. To test this, we perform an OLS regression with **shop-level sentiment** as the dependent variable **y** and **average shop rating** as the independent variable **x**.
    ''')
st.latex(r'AverageSentiment_i = \beta_0 + \beta_1 * AverageRating_i')
st.latex(r'H_0: \beta_1 = 0; H_1: \beta_1 \neq 0')

st.write('''
2. Comments can be grouped into various topics, such as flavor, product variety, or service attitude. 
         
    To investigate this, we apply topic modeling techniques like **k-means** and **LDA**, along with keyword analysis tools such as **word clouds**, to qualitatively explore this idea.
''')

st.write('''
3. Comments exhibit distinct characteristics across different brands.  

   To explore this, we conduct keyword analysis for each brand and qualitatively identify features that align with our daily observations.
''')

st.write('''
4. The distance to MRT station has negative impact on total comment (or rating) counts:
    
    We assume that people tend to consume in the shops near stations, so we expect to observe a negative relationship between MRT station and total comment (or rating) counts.
''')
st.latex(r'TotalCount_i = \beta_0 + \beta_1 * Dist_i')
st.latex(r'H_0: \beta_1 = 0; H_1: \beta_1 < 0')
st.write('''
5. Proximity to MRT stations positively impacts average ratings:  

   We assume that the closer a shop is to an MRT station, the higher the consumer traffic. Consequently, an increase in the number of consumers may reasonably lead to a rise in negative scores.
''')
st.latex(r'AverageRating_i = \beta_0 + \beta_1 * Dist_i')
st.latex(r'H_0: \beta_1 = 0; H_1: \beta_1 > 0')
st.divider()




# *** Data Source
st.header("Data Source", anchor = "Data_Source")
st.write("""
Regarding the data source, we focus on bubble tea shops located near MRT stations in Taipei City and New Taipei City. By **near**, we refer to shops situated within a 1-kilometer radius of each MRT station.

- Google Map API:
         
    We start with collecting shop level data. We query all bubble tea shops that fulfill our standard (within 1-kilometer radius of MRT station) by **Google Map API** to get these shops' **id**, **longitude**, and **latitude**.
- Webscraping:
         
    Using the complete dataset of shops, we can collect comment data through **web scraping** using the Selenium package in Python. For each shop, we send requests to open the Google Maps page by embedding the shop's **ID** into the URL, set the comment order to "most recent," scroll down 30 times, and then extract all available comment texts from the HTML source code.""")

st.divider()

# *** Methodology
st.header("Methodology", anchor = "Methodology")
st.write("""
- Tokenization
- Topic Modeling (LDA / K-means)
- Sentiment Analysis (KeyMoji)
- Wordcloud
- Simple Linear Regression (OLS)
""")

st.divider()

# **************************************** Results ****************************************
st.header("Results", anchor = "Results")

# *** average rating & average sentiment
st.markdown("<h4> Average rating & Average sentiment</h4>", unsafe_allow_html = True)

st.write('''
In order to verify our main hypothesis, we conduct ordinary least square regression for the rescaled average sentiment scores on the rating score across shop. The sentiment scores is calculated for all comments then grouped by "shop id" to calculate the average as the proxy of shop level scores; the rating score is directly scraped with **selenium** package from google review. To ensure the robustness, we have removed outliers for both variables using IQR method with multiplier 1.5.''')


reg1_table, reg1_plot = st.columns((0.45, 0.55))

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

Overall, the average rating generally aligns with consumers' perceptions towards the shops. However, a significant number of shops still deviate considerably from the regression line. **To identify these shops, we define observations with high sentiment score and low average rating (or conversely, low sentiment score and high average rating) as "inconsistent" observations.** More specifically, we define the status of inconsistenty as:
''')

st.latex(r'IC_i = \mathbb{I}((R_i - Q_{\frac{1}{2}}(R_i)) \cdot (S_i - Q_{\frac{1}{2}}(S_i)) < 0)')

st.markdown(r"where *$R_i$* represents the average rating of shop i, *$S_i$* represents the average sentiment of shop i, *$Q_{\frac{1}{2}}(\cdot)$* being the **median** function, and *$\mathbb{I}(\cdot)$* being the indicator function.")

# ***** Example of Inconsistent Shop
incons_u, cons_u = st.columns((0.5, 0.5))
incons_d, cons_d = st.columns((0.5, 0.5))
with incons_u:
    text = " ".join(comments[comments['name'].str.contains('50嵐 淡水')]['processed_comments']).replace(" ", ",")
    stopwords = [
                '店家', '味道', '飲料', '店員', '真的', '點餐', '客人', '一杯', '點了', '很多', '發現', 'nan', '每次', '服務態度', '服務', '態度', '一個', '清心', '現場', '分鐘', '感覺', '飲品', '推薦', "珍珠", "大杯", '紅茶', '中杯', '北投', '黑糖', '評論', '甜度', '無糖', '士林', '巴士', '爆打', '夜市', '荔枝', '檸檬', '時間', '奶茶', '四季春', '五十', '吸管', '波霸', '萬波'
            ]

    for word in stopwords:
        text = text.replace(f"{word}", "")

    # Create and generate a word cloud image:
    with st.spinner("Loading wordclouds..."):
        wordcloud = WordCloud(
            background_color=None,  # No background color
            mode='RGBA',             # Enable transparency
            font_path='./font.ttf',
            random_state = 1214,
            width = 500,
            height = 300
            # mask = mask
        ).generate(text)


        fig, ax = plt.subplots(1, 1)
        ax.imshow(wordcloud, interpolation = 'bilinear')
        ax.axis('off')
        ax.set_title("Example wordcloud of inconsistent shop", color = "gray")
        fig.patch.set_alpha(0)
        st.pyplot(fig)


with incons_d:
    st.write("""We pick several inconsistent shops that deviate the most from the regression line and then visualize their comments using word clouds to explore qualitative evidence for the source of inconsistency. 
    
One distinctive example is 50嵐 淡水英專店 (marked as red dot in the scatter plot above), which has a relatively low **average rating of 3.6**, yet a high **average sentiment score of 3.495** is also observed. The wordcloud also shows that it has high frequency of posivite words, such as **好喝, 親切, 很棒**.""")
    
# ***** Example of Consistent Shop
with cons_u:
    text = " ".join(comments[comments['name'].str.contains('Tea Shop 台北大安')]['processed_comments']).replace(" ", ",")
    stopwords = [
                '店家', '味道', '飲料', '店員', '真的', '點餐', '客人', '一杯', '點了', '很多', '發現', 'nan', '每次', '服務態度', '服務', '態度', '一個', '清心', '現場', '分鐘', '感覺', '飲品', '推薦', "珍珠", "大杯", '紅茶', '中杯', '北投', '黑糖', '評論', '甜度', '無糖', '士林', '巴士', '爆打', '夜市', '荔枝', '檸檬', '時間', '奶茶', '四季春', '五十', '吸管', '波霸', '萬波', '喝過', '這家'
            ]

    for word in stopwords:
        text = text.replace(f"{word}", "")

    # Create and generate a word cloud image:
    with st.spinner("Loading wordclouds..."):
        wordcloud = WordCloud(
            background_color=None,  # No background color
            mode='RGBA',             # Enable transparency
            font_path='./font.ttf',
            random_state = 1214,
            width = 500,
            height = 300
            # mask = mask
        ).generate(text)


        fig, ax = plt.subplots(1, 1)
        ax.imshow(wordcloud, interpolation = 'bilinear')
        ax.axis('off')
        ax.set_title("Example wordcloud of consistent shop", color = "gray")
        fig.patch.set_alpha(0)
        st.pyplot(fig)

with cons_d:
    st.write("""On the other hand, we also analyze examples for consistent shops. 
             
An intriguing example is **萬波島嶼紅茶 台北大安店**, which stands out with both a low average rating **(2.8)** and a low average sentiment score **(2.55)**. Consistent with these ratings, the word cloud reveals a high frequency of terms like **"糟糕"** (terrible) and **"收一收"** (having pessimistic view about the shop's future), accurately reflecting its poor performance in Google Maps reviews.


""")
# ----

st.write(rf"""
Ovarall, the proportion of inconsistent shops is **{round(sum(shops['inconsistent'] == 'True')/len(shops['inconsistent']), 4)}**, and the proportion of inconsistent shops conditional on identified brand is plotted as follow.
""")

brands = brands[brands['shop count'] > 10]
bar_incons = go.Bar(
    name = 'Inconsistent',
    x = brands["brand"],
    y = brands["inconsistent"],
    offsetgroup = 0,
    text = brands["incons_prop"]
)

bar_cons = go.Bar(
    name = 'Consistent',
    x = brands["brand"],
    y = brands["consistent"],
    base = brands["inconsistent"],
    offsetgroup = 0,
    text = brands["cons_prop"]
)

fig_brand_incons = go.Figure(
    data = [bar_incons, bar_cons],
    layout = go.Layout(
        title = f"% of inconsistent shops (height of bars: shop counts)"
    ))

fig_brand_incons.update_traces(textposition = 'inside')
st.plotly_chart(fig_brand_incons)

st.write("""We focus exclusively on brands with a sample size (number of shops) exceeding a predefined threshold of 10. Among these, **50嵐**, one of Taiwan's largest bubble tea brands, stands out for its low inconsistency rate (23%) with the highest shop counts, drawing our attention. Our research reveals that 50嵐 has adopted a direct-selling-focused strategy and maintains stringent standards for granting franchise licenses. Historically, only staff with over a year of experience in any branch were prioritized for franchise opportunities. Currently, the brand has ceased all franchising and operates as a fully direct-selling brand in Taiwan. This approach ensures consistent quality across branches, which may explain its relatively low inconsistency rate.
""")


        

# *** Topic Modeling & Wordcloud
st.markdown("<h4> Topic Modeling & Wordcloud</h4>", unsafe_allow_html=True)

st.write("""
To gain deeper insights, we applied topic modeling algorithms to identify the key aspects people focus on when discussing bubble tea shops. Both LDA and k-means clustering were tested with \( k = 3 \) and \( k = 4 \). After manual evaluation, the optimal results were achieved using k-means with \( k = 3 \). Considering that people might leave comments with mixed aspects, we allow each comment to be assigned with 2 topics. Finally, we visualize the results by **topic-conditional wordclouds** as follow.
""")

# comments_filtered_by_dims = comments.loc[comments['category'].isin(dims), :].dropna()
wc_l, wc_m, wc_r = st.columns(3)
with st.spinner("Loading wordclouds..."):
    for i, (placeholder, dim) in enumerate(zip([wc_l, wc_m, wc_r], ['品項', '口味', '服務態度'])):
        comments_filtered_by_dims = comments.loc[comments['category'].isin([dim]), :].dropna()
        with placeholder:
            fig = PlotManager.worcdloud_generate(comments_filtered_by_dims, 'All')[1]
            st.pyplot(fig)
            st.caption(f"Topic {i + 1}")

st.write('''As shown, topic 1 includes keywords such as "珍珠," "奶茶," "紅茶," and "奶蓋," which can be identified as **"product diversity (品項)"**. On the other hand, topics 2 and 3 share a similar concept, both featuring "親切" with high frequency. These two topics are not clearly distinguishable, and we can only conclude that they are associated with "service quality (服務態度)".

The challenge likely stems from the fact that people often leave comments containing mixed concepts. Intuitively, a single comment rarely focuses on just one topic. For instance, someone might praise a shop's drinks while criticizing the service. This inherent nature makes it extremely difficult to achieve clear clustering results. Another contributing factor could be the brevity of the comments and the limited language variety used in bubble tea reviews. When most comments are short and feature similar vocabulary, categorizing them becomes a significant challenge.
        
For a more insightful analysis, we merge topics 2 and 3 and visualize them again. The results provide clearer insights: (new) topic 1 focuses on products and toppings (e.g., 推薦, 排雷), while (new) topic 2 emphasizes service, waiting time, and staff attitude.''')

wc_new_l, wc_new_r = st.columns(2)
with st.spinner("Loading wordclouds..."):
    for i, (placeholder, dim_lst) in enumerate(zip([wc_new_l, wc_new_r], [['品項'], ['口味', '服務態度']])):
        comments_filtered_by_dims = comments.loc[comments['category'].isin(dim_lst), :].dropna()
        with placeholder:
            fig = PlotManager.worcdloud_generate(comments_filtered_by_dims, 'All', 300, 200)[1]
            st.pyplot(fig)
            st.caption(f"(new) Topic {i + 1}")

st.write("""
Overall, we conclude that regarding products, "珍珠" (bubble), "奶茶" (milk tea), and "紅茶" (black tea) are the three most prominent items for Taiwanese consumers, followed by "烏龍" (Oolong) and "奶蓋" (cheese foam), which are also popular choices. On the service side, consumers frequently mention "親切" (friendly), suggesting that a welcoming attitude from staff can leave a positive impression and enhance brand favorability.
""")

# *** brand level wordcloud
st.markdown("<h4>Brand level Keyword analysis</h4>", unsafe_allow_html=True)
st.write("""
We are also interested in what is being discussed for each brand. Therefore, we conduct **keyword analysis** for some brands of interest. 
        
- 50嵐 has a high frequency of the words "珍珠" (small bubbles) and "波霸" (big bubbles), suggesting that the shop is particularly known for its quality bubbles. In addition, "親切" also accounts significantly, which implies its service attitude might be above level.
        
- 龜記 is best known for its "紅柚翡翠" (grapefruit-flavored green tea), which stands out as its signature product and has driven a surge in sales. (It’s also my personal favorite!)
        
- 得正 shows the highest frequency for the word "烏龍" (Oolong), which aligns with its market positioning as it initially focused on Oolong tea. The word "排隊" (waiting in line) also appears frequently, consistent with our personal observations—there’s always a queue whenever we buy bubble tea from 得正.
""")

brand = st.selectbox("Choose the brand", st.session_state['brands'] + ['All'], index = 13)
wc = PlotManager.worcdloud_generate(comments, brand)
df_wordcloud = wc[0]
df_wordcloud = df_wordcloud[df_wordcloud['count'] > df_wordcloud['count'].quantile(0.9)].sort_values('count', ascending = True)
l, r = st.columns((0.6, 0.4))
with l:
    fig = px.bar(df_wordcloud, 'count', 'word', orientation = 'h', height = 480)
    st.plotly_chart(fig)
with r:
    st.pyplot(wc[1])


# *** distance & comment / rating counts
st.markdown("<h4>Distance to the nearest MRT exit & Comment Counts</h4>", unsafe_allow_html=True)

st.write("The scatterplot and regression analysis reveal a weak but statistically significant negative relationship between the distance to the nearest MRT exit and the total number of comments or ratings for bubble tea shops. The R-squared value of 0.023 indicates that the model explains only 2.3% of the variability in comment counts. The coefficient for distance to MRT is -0.1118 (p-value = 0.003), suggesting that for every 1-kilometer increase in distance from an MRT exit, the number of comments decreases by approximately 11. This result highlights that shops located closer to MRT exits tend to receive more customer engagement, as reflected by higher comment counts.")

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

st.write("The analysis shows a slight but statistically significant positive correlation between the distance to the nearest MRT exit and the average rating of bubble tea shops. With an R-squared value of 0.030, the model captures just 3% of the variation in average ratings. The coefficient of 0.0004 (p-value = 0.001) suggests that for every additional meter away from an MRT exit, the average rating increases marginally. While the impact is minimal, this could indicate that shops located further from MRT stations tend to receive slightly better ratings.")

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



# *** brand average scores
# shops = ConfigManager.get_shop_data()
# brands = ConfigManager.get_brand_data(shops)
# fig = px.bar(brands, 'brand', '服務態度_score', hover_data = ['shop count'])
# fig.update_traces(marker = dict(color = '#E6DCB9'))
# st.plotly_chart(fig)
    


# *** raw data tables
with st.expander('Raw Data'):
    with st.container(key = 'dfs'):
        st.markdown("<h3 style='text-align: center; '>Shops</h3>", unsafe_allow_html=True)
        st.write(shops)

        st.markdown("<h3 style='text-align: center; '>Comments</h3>", unsafe_allow_html=True)
        st.write(comments)

        st.markdown("<h3 style='text-align: center; '>Brands</h3>", unsafe_allow_html=True)
        st.write(brands)