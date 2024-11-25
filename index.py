import streamlit as st

st.set_page_config(layout = 'wide')

pgs = st.navigation(
    pages = [
        st.Page('report.py', title = 'Report'),
        st.Page('dashboard.py', title = 'Dashboard')
    ]
)

pgs.run()

