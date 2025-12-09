__We recommend using PyCharm as the IDE, as it is the platform on which all code was created.__  

__Scraping Data:__  
Ensure the most recent versions of os, re, requests, pandas, and io are installed

All csv files needed to run main.py have been provided.
However, if the user wishes to test and run the scraper to obtain the 
csv files themselves, they must run run_scraper.py. This file
will scrape and clean the necessary data, adding them to folder titled cleaned_data.
Manual changes must then be made to this data to account for missing years and inconsistencies. 
This is done by copying and pasting the necessary additions into these proper rows. Rows must be
placed between the years that surround them (ex. if a 1943-1944 season is missing, paste the missing
row above the 1942-1943 season and below the 1944-1945 season).

Additions are as follows.  
  
__football.csv:__  
1943-1944,0,0,0.0,no

__basketball.csv:__  
1987-1988,23.0,12.0,65.714,no  
1986-1987,23.0,11.0,67.647,no  

1943-1944,0,0,0.0,no  

__baseball.csv:__  
205-2025,0,0,0.0,no  

1943-1944,0,0,0.0,no  
1942-1943,0,0,0.0,no  

__soccer.csv:__  
2025-2026,6.0,7.0,46.154,no  
2024-2025,4.0,8.0,33.333,no  
2023-2024,6.0,5.0,54.545,no  
2022-2023,2.0,14.0,12.500,no  
2021-2022,4.0,12.0,25.0,no  
2020-2021,6.0,8.0,42.857,no  
2019-2020,11.0,9.0,52.380,no  

__tennis.csv:__  
2025-2026,0,0,0.0,no  
2024-2025,15,12,55.556,no  
2023-2023,13,12,52.0,no  
2022-2023,14,14,50.0,no  

__womens_tennis.csv:__  
2025-2026,0,0,0.0,no  
2024-2025,9,14,39.130,no  
2023-2024,17,9,65.385,no  
2022-2023,18,8,69.231,no  
2021-2022,21,7,75.0,no  
2020-2021,13,8,61.905,no  
2019-2020,5,4,55.556,no  
2018-2019,13,12,52.0,no  
2017-2018,19,9,67.857,no  
2016-2017,29,3,90.625,yes  
2015-2016,23,3,88.462,no  
2014-2015,24,4,85.714,no  
2013-2014,23,6,79.310,no  
2012-2013,26,3,89.655,no  

__softball.csv:__  
2025-2026,0,0,0.0,no

The other csv files (volleyball.csv and womens_basketball.csv) do not require additional editing.

__Running main.py:__ 
Ensure the most recent versions of pandas, numpy, and matplotlib.pyplot are installed.  

In order to run main.py and user the interface, the user must simply run the code on their chosen IDE (PyCharm is
recommended). The code prompts the user on how to use it in the console, so follow instructions as directed.
