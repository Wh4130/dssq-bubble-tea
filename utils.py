import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import numpy as np
from wordcloud import WordCloud
import random
from streamlit_folium import st_folium
import folium

class ConfigManager:

    @st.cache_data()
    def generate_low_saturation_colors(n = 50):
        """
        Generate `n` low-saturation colors in RGB format.

        Parameters:
            n (int): Number of colors to generate.

        Returns:
            list: List of RGB tuples representing low-saturation colors.
        """
        colors_hex = []
        for _ in range(n):
            hue = random.random()  # Random hue (0-1)
            saturation = random.uniform(0.1, 0.4)  # Low saturation (10-40%)
            lightness = random.uniform(0.5, 0.8)  # Medium lightness (50-80%)
            
            # Convert HSL to RGB
            import colorsys
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            # Convert RGB to HEX
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            colors_hex.append(hex_color)
        return colors_hex
    
    @st.cache_data
    def get_shop_data():
        shops = pd.read_excel('./files/processed_shops_info.xlsx').drop('Unnamed: 0', axis = 1)
        return shops

    @st.cache_data()
    def get_comment_data():
        df = pd.read_excel('./files/processed_comments_with_shops.xlsx').drop('Unnamed: 0', axis = 1)
        df['processed_comments'] = df['processed_comments'].astype(str)
        df['text'] = df['text'].astype(str)
        return df
    
    @st.cache_data
    def get_brand_data(shop_data):
        brands =  shop_data.groupby('brand').agg({'name': 'count',
                                        'average_rating': 'mean',
                                        '品項_score': 'mean',
                                        '口味_score': 'mean',
                                        '服務態度_score': 'mean',
                                        'avg_sentiment': 'mean'}).reset_index().rename(columns = {'name': 'shop count'})
        brands = brands.loc[brands['brand'] != 'Other']
        return brands
    
    @st.cache_data
    def get_mrt_data():
        df = pd.read_excel('./files/mrt_stations.xlsx', engine = 'openpyxl')
        df = df.rename(columns = {
            '出入口名稱': 'name_exit',
            '出入口編號': 'exit',
            '經度': 'longitude',
            '緯度': 'latitude', 
        }).drop(['是否為無障礙用', '項次'], axis = 1)
        return df

    def significant_brands() -> list:
        return ["CoCo", "Other", "珍煮丹", "鶴茶樓", 
                "清心福全", "迷客夏", "功夫茶", "得正", 
                "武林茶", "再睡5分鐘", "茶聚", "COMEBUY", 
                "天仁茗茶", "50嵐", "萬波", "可不可熟成紅茶", 
                "龜記", "麻古茶坊"] 

    def brand_color_mapping() -> dict:
        return {x: y for x, y in zip(ConfigManager.significant_brands() + ['All'], ConfigManager.generate_low_saturation_colors(20))}
    
    def metric_explanation() -> dict:
        return {
            'average_rating': 'The average rating of each brand scraped from google comments',
            '品項_score': 'The sentiment score of each brand for "product diversity" dimension',
            '口味_score': 'The sentiment score of each brand for "flavor" dimension',
            '服務態度_score': 'The sentiment score of each brand for "service attitude" dimension',
            'avg_sentiment': 'The average of "品項 score", "口味 score", "服務態度_score"',
            'shop count': 'Total number of scraped shops for each brand'


        }


class PlotManager:
    @staticmethod
    def brands_barplot(brands, metric):
        fig = px.bar(brands, 'brand', metric, hover_data = ['shop count'])
        fig.update_traces(marker = dict(color = "#E6DCB9"))
        return fig
    
    @staticmethod
    def rating_regplot(shops, brand, title):
        if (brand == 'All'):
            data = shops
            fig = px.scatter(
                data,
                'average_rating',
                'avg_sentiment',
                trendline = 'ols',
                title = title,
                hover_name = 'name'
            )
        else:
            data = shops[shops['brand'] == brand]
            fig = px.scatter(
                data,
                'average_rating',
                'avg_sentiment',
                trendline = 'ols',
                title = title,
                hover_name = 'name'
            )
        fig.update_traces(marker = dict(size = 8, color = st.session_state['brand_color_mapping'][brand]))
        fig.update_layout(title_x = 0.45)

        return fig
    
    @staticmethod
    def worcdloud_generate(data, brand):

        x, y = np.ogrid[:300, :300]
        mask = (x - 150)**2 + (y - 150)**2 > 150**2
        mask = 255 * mask.astype(int)
        '''
        text should be separated by comma
        '''
        if brand == 'All':
            text = ' '.join(data.loc[:, 'processed_comments'].astype(str)).replace(' ', ', ')
        else:
            text = ' '.join(data.loc[data['brand'] == brand, 'processed_comments'].astype(str)).replace(' ', ', ')

        stopwords = [
            '店家', '味道', '好喝', '飲料', '店員', '真的', '點餐', '客人', '一杯', '點了', '很多', '發現', 'nan', '每次', '服務態度', '服務', '態度', '一個',
            brand
        ]


        for word in stopwords:
            text = text.replace(f"{word}, ", "")



        # Create and generate a word cloud image:
        wordcloud = WordCloud(
            background_color=None,  # No background color
            mode='RGBA',             # Enable transparency
            font_path='./font.ttf',
            random_state = 000,
            width = 300,
            height = 400
            # mask = mask
        ).generate(text)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(wordcloud, interpolation = 'bilinear')
        ax.axis('off')
        fig.patch.set_alpha(0)

        return fig
    
    @staticmethod
    def random_pick_comment(df, brand):
    
        if brand == 'All':
            pass
        else:
            df = df.loc[df['brand'] == brand]

        df = df.reset_index(drop=True)
        idx = random.randint(0, len(df) - 1)
        picked = df.loc[idx, ['text', 'name']]
        return f"{picked['text']} ({picked['name']})"
    #  f"{picked['text'] (picked['name'])}"

    @staticmethod
    @st.cache_data
    def init_map(mrt_stations):

        m = folium.Map(location = [mrt_stations['latitude'].mean(), mrt_stations['longitude'].mean()], width = '100%', height = '100%', zoom_start = 11)
        
        for index, row in mrt_stations.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                stroke=False,
                radius=3.2,
                fill=True,
                fill_opacity=0.6,
                fill_color='black',
                tooltip=folium.Tooltip(
                    f"{row['name_exit']}", style = f"color:{row['line']}",  sticky = True)
            ).add_to(m)

        return m

    @staticmethod
    @st.cache_data
    def add_shops_to_map(shops, brand):
        pass
        # ** filter 
        # ** use brand-color map to determine colors
        
