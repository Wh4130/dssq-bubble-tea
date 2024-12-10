# Taiwanese Drink Shop Free Research

#### Link to our streamlit app
https://dssq-bubble-tea-m9jzmospevfsw8uh2ncz4o.streamlit.app/

## Files
### `report.py` (UI)
This page contains the complete content of our research, including motivation, data collection, methodology, and results.
### `dashboard.py` (UI)
This page is an interactive dashboard that allows user to compare average ratings, average sentiment score of comments, and what people comment about for two selected brands. The geographic distribution and raw data are also displayed here.
### `index.py` (UI)
This page is the entrance of our application. 
### `utils.py` (back end)
This page contains a series of functions that would be used in UI pages. Functions are defined under two classes with static method to improve the program's architecture.

- Config Manager: functions that access  manipulate data are defined under this class.
- Plot Manager: functions that manages visualization and return `fig` object are defined here.