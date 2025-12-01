import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sb_df = pd.read_csv('softball.csv')
bb_df = pd.read_csv('baseball.csv')
fb_df = pd.read_csv('football.csv')
ten_df = pd.read_csv('tennis.csv')
wten_df = pd.read_csv('womens_tennis.csv')
bk_df = pd.read_csv('basketball.csv')
wbk_df = pd.read_csv('womens_basketball.csv')
soc_df = pd.read_csv('soccer.csv')
vb_df = pd.read_csv('volleyball.csv')

def check_sport_validity(sport):
    if (sport!= "football" and sport!= "basketball" and sport!= "baseball" and sport!= "softball" and sport!= "women's basketball" and sport!= "women's tennis" and sport!= "soccer"  and sport!= "volleyball" and sport != 'tennis'):
        return False
    return True

def retrieve_data_frame(sport):
    if sport == 'softball':
        return sb_df
    elif sport == 'baseball':
       return bb_df
    #uncomment as dataframes become available
    elif sport == 'football':
       return fb_df
    #elif sport == 'tennis':
       #return ten_df
    elif sport == "women's tennis":
       return wten_df
    elif sport == 'basketball':
        return bk_df
    elif sport == "women's basketball":
        return wbk_df
    elif sport == 'soccer':
        return soc_df
    elif sport == "volleyball":
        return vb_df

def convert_year_to_integer(year_string):
    start = year_string.split("-")[0]
    return int(start)

def get_color(sport):
    if sport == 'softball':
        return 'olive'
    elif sport == 'baseball':
        return 'green'
    elif sport == 'football':
        return 'orange'
    elif sport == 'tennis':
        return 'red'
    elif sport == "women's tennis":
        return 'pink'
    elif sport == 'basketball':
        return 'blue'
    elif sport == "women's basketball":
        return 'cyan'
    elif sport == 'soccer':
        return 'gray'
    elif sport == "volleyball":
        return 'purple'

#not done
def comparing_all_sports():
    return 0

#change colors
def stacked_bar_plot():
    xticks = [0]
    xticklabels = ['1925-1926']
    temp_sport_list = ['baseball','softball',"women's basketball",'volleyball','basketball','football',"women's tennis",'soccer']
    start_year = 1926

    base_list = ['1925-1926']
    columns = ['season']
    for i in range(0,len(temp_sport_list)):
        sport_df = retrieve_data_frame(temp_sport_list[i])
        base_list.append(0)
        columns.append(temp_sport_list[i])
    df = pd.DataFrame([base_list], columns=columns)
    num_rows = 1

    for i in range(2,len(retrieve_data_frame('football')['national_championship'].tolist())+1):
        vals = [retrieve_data_frame('football')['year'].iloc[-i]]
        if int(retrieve_data_frame('football')['year'].iloc[-i][3])==5:
            xticks.append(i-1)
            xticklabels.append(retrieve_data_frame('football')['year'].iloc[-i])
        for j in range(0,len(temp_sport_list)):
            sport_df = retrieve_data_frame(temp_sport_list[j])
            df_start_year = convert_year_to_integer(sport_df['year'].iloc[-1])
            if df_start_year > start_year:
                if i <= df_start_year - start_year:
                    vals.append(0)
                else:
                    if retrieve_data_frame(temp_sport_list[j])['national_championship'].iloc[-i+(convert_year_to_integer(sport_df['year'].iloc[-1]) - start_year)+1] == 'yes':
                        vals.append(1)
                    else:
                        vals.append(0)
            elif retrieve_data_frame(temp_sport_list[j])['national_championship'].iloc[-i] == 'yes':
                vals.append(1)
            else:
                vals.append(0)
        df.loc[num_rows] = vals
        num_rows += 1

    df.plot(kind='bar', stacked=True)
    plt.xlabel('Season', fontsize=10)
    plt.ylabel('Number of Championship', fontsize=10)
    plt.yticks([0,1,2],[0,1,2])
    plt.title('Championship Wins per Year Over 100 Years')
    plt.xticks(xticks,xticklabels,fontsize=6, rotation=0)
    plt.show()
    #['1925-1926','1930-1931','1940-1941','1950-1951','1960-1961','1970-1971','1980-1981','1990-1991','2000-2001','2010-2011','2020-2021','2025-2026']


#option 1 graph function, championship comparison
def champ_sports_comparison(champ_sport):
    sport_list = ["baseball","softball","women's basketball","volleyball","basketball","football","women's tennis","soccer"]

    sports_df = {}
    for sport in sport_list:
        df = retrieve_data_frame(sport)
        df["start_year"] = df["year"].apply(convert_year_to_integer)
        df["season_label"] = df["year"]
        sports_df[sport] = df

    #get championship years from the selected sport
    champion_df = sports_df[champ_sport]
    championship_years = set(champion_df.loc[champion_df["national_championship"] == "yes", "start_year"])
    if not championship_years:
        print(f"{champ_sport.capitalize()} has no championship seasons in this dataset.")
        return

    #filters years of all other sports to match championship years
    comparison_dfs = {}
    for sport, df in sports_df.items():
        if sport != champ_sport:
            filtered = df[df["start_year"].isin(championship_years)]
            comparison_dfs[sport] = filtered.sort_values("start_year")

    #plot
    plt.figure(figsize=(10, 8))

    #uses given championship sport for x-axis
    champ_df = sports_df[champ_sport]
    champ_df = champ_df[champ_df["start_year"].isin(championship_years)]
    champ_df = champ_df.sort_values("start_year")
    x_labels = list(champ_df["season_label"])
    x_positions = list(range(len(x_labels)))

    #plots all other sports' win%
    year_to_pos = {}
    position = 0
    for year in champ_df["start_year"]:
        year_to_pos[year] = position
        position += 1

    # Plot each sport
    for sport, df in comparison_dfs.items():
        years = []
        for year in df["start_year"]:
            years.append(year_to_pos[year])

        win_pct = df["win_loss_pct"].tolist()
        plt.scatter(years, win_pct, s=120, color = get_color(sport), label=f"{sport.capitalize()} Win %")

    plt.xticks(x_positions, x_labels)
    plt.xlabel(f"{champ_sport.capitalize()} Championship Season")
    plt.ylabel("Win Percentage")
    plt.title(f"Win Percentage Across UF Sports During {champ_sport.capitalize()} Championship Seasons")
    plt.legend()
    plt.show()


#make sure r analysis works, option 3 function
def sports_correlation(sport1, sport2):
    sport1_df = retrieve_data_frame(sport1)
    sport2_df = retrieve_data_frame(sport2)

    #Converts each sport's 'year' string to an integer and stores as a new column
    integer_years_df1 = []
    for season in sport1_df["year"]:
        converted = convert_year_to_integer(season)
        integer_years_df1.append(converted)
    sport1_df["integer_years"] = integer_years_df1

    integer_years_df2 = []
    for season in sport2_df["year"]:
        converted = convert_year_to_integer(season)
        integer_years_df2.append(converted)
    sport2_df["integer_years"] = integer_years_df2

    #sorts both sports chronologically using the integer years
    sport1_df = sport1_df.sort_values("integer_years")
    sport2_df = sport2_df.sort_values("integer_years")

    #use sets to find overlapping years
    common_years = set(sport1_df["integer_years"]).intersection(set(sport2_df["integer_years"]))

    #filters both sports to only overlapping years
    sport1_df = sport1_df[sport1_df["integer_years"].isin(common_years)]
    sport2_df = sport2_df[sport2_df["integer_years"].isin(common_years)]

    #resets both sports indecies so they align when graphing
    sport1_df = sport1_df.reset_index(drop=True)
    sport2_df = sport2_df.reset_index(drop=True)

    #extracts lists with each sports' win% over the years
    win1 = list(sport1_df["win_loss_pct"])
    win2 = list(sport2_df["win_loss_pct"])

    #converts to numpy arrays to be plotted
    x = np.array(win1)
    y = np.array(win2)

    #computes correlation coefficient
    r = np.corrcoef(x,y)[0,1]
    #computes line of best fit
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    m = np.sum((x - mean_x)*(y - mean_y)) / np.sum((x - mean_x)**2)
    b = mean_y - (m * mean_x)
    best_fit = m * x + b

    plt.figure(figsize=(10,10)) #we can change this, I just put in a random scale for now
    plt.scatter(x, y, color = "blue", label = "Win Percentage Data Points")
    plt.plot(x, best_fit, color = "red", label = "Line of Best Fit")


    plt.xlabel(f"{sport1.capitalize()} Win Percentage")
    plt.ylabel(f"{sport2.capitalize()} Win Percentage")
    plt.title(f"{sport1.capitalize()} vs. {sport2.capitalize()}\n Correlation Coefficient: {r}")
    plt.legend()
    plt.show()

    #analyzing correlation coefficient
    if r == 0:
        print(f"The win percentages over time of {sport1.capitalize()} and {sport2.capitalize()} have no correlation.")
    else:
        if r < 0:
            direction = "negative"
        elif r > 0:
            direction = "positive"

        if abs(r) >= .7 and abs(r) <= 1:
            strength = "very string"
        elif abs(r) >= .5 and abs(r) < .7:
            strength = "strong"
        elif abs(r) >= .3 and abs(r) < .5:
            strength = "moderate"
        elif abs(r) > 0 and abs(r) < .3:
            strength = "weak"

        print(f"With a correlation coefficient of {r:.2f}, the win percentages over time of {sport1.capitalize()} and {sport2.capitalize()} have a {strength}, {direction} correlation.")


#ensure it works when csvs are correct, option 2 function
def compare_sports_means(sport_list):
    #loads all df into a dictionary
    sports_df = {}
    for sport in sport_list:
        df = retrieve_data_frame(sport)
        df["start_year"] = df["year"].apply(convert_year_to_integer)
        df["season_label"] = df["year"]
        sports_df[sport] = df

    #determines valid year range
    all_years = set()
    for df in sports_df.values():
        all_years.update(df["start_year"])
    min_year = min(all_years)
    max_year = max(all_years)

    #user selects interval to analyze over
    try:
        start_year_input = int(input("Enter the year you wish to start the comparison: "))
        end_year_input = int(input("Enter the year you wish to end the comparison: "))

        if start_year_input < min_year or end_year_input > max_year:
            print(f"Years must be between {min_year} and {max_year}.")
            return

        if start_year_input > end_year_input:
            print("Start year must be before end year.")
            return

    except ValueError:
        print("Please enter valid integer years.")
        return

    # Filters dfs to selected interval
    sport_interval_dfs = {}
    for sport, df in sports_df.items():
        interval_df = df[(df["start_year"] >= start_year_input) & (df["start_year"] <= end_year_input)]
        sport_interval_dfs[sport] = interval_df

    # Determine overlapping seasons
    year_sets = []
    for df in sport_interval_dfs.values():
        year_sets.append(set(df["start_year"]))
    shared_years = sorted(set.intersection(*year_sets))

    if not shared_years:
        print("There are no overlapping seasons between these sports in this interval.")
        return

    #ensures only shared years are being evaluated
    filtered_df = {}
    interval_df = {}
    for sport, df in sports_df.items():
        filtered_df[sport] = df[df["start_year"].isin(shared_years)]
        sorted_df = filtered_df[sport].sort_values("start_year")
        interval_df[sport] = sorted_df

    #plot graph
    plt.figure(figsize=(10, 10))

    #basic info needed for plotting (years, win%s, labels)
    for sport in sport_list:
        df = interval_df[sport]
        years = list(df["start_year"])
        win_percentages = list(df["win_loss_pct"])
        labels = list(df["season_label"])

        plt.plot(years, win_percentages, linewidth = 3, marker = "o", color = get_color(sport), label = f"{sport.capitalize()} Win Percentage")

        #analysis of total change over the interval
        start_win_percentages = win_percentages[0]
        end_win_percentages = win_percentages[-1]
        total_change = end_win_percentages - start_win_percentages

        #analysis of how much win% changed per year on average (mean)
        average_change = total_change / (len(win_percentages)-1)

        if total_change > 0:
            direction = "increased"
        elif total_change < 0:
            direction = "decreased"
        else:
            direction = "stayed the same"

        print(f"{sport.capitalize()}'s win percentage {direction} by {abs(total_change):.2f} points from the {labels[0]} season to the {labels[-1]} season.")
        print(f"{sport.capitalize()}'s win percentage {direction} by {abs(average_change):.2f} on average from year to year.")

    #labels for x-axis
    example_sport = sport_list[0]
    plt.xticks(interval_df[example_sport]["start_year"], interval_df[example_sport]["season_label"])

    #title
    sports_names_formatted = ", ".join([sport.capitalize() for sport in sport_list])
    labels = list(df["season_label"])
    year_range = f"{labels[0]} Season to {labels[-1]} Season"
    plt.title(f"Performance Trends: {sports_names_formatted}\n{year_range}")
    plt.xlabel("Season")
    plt.ylabel("Win Percentage")

    plt.legend()
    plt.show()

stacked_bar_plot()
print("Welcome to the Chomp Chain Reaction Data Analyzer!")
running = True
while running:
    print("1. Compare the win percentages of all sports during a specific sport's championship seasons")
    print("2. Compare the average rate of change and total change of sports' win percentages over a specific interval")
    print("3. Find the correlation between two sports' win percentages")
    print("4. Exit")
    option = input("Type in a number to select an option: ")

    if option == '1':
        champ_sport = input("Enter the championship sport for comparison: ")
        champ_sport = champ_sport.lower()
        if check_sport_validity(champ_sport) == False:
            print("Please enter a valid sport.")
            print("")
            continue
        champ_sports_comparison(champ_sport)

    elif option == '2':
        num_sports = int(input("Enter the number of sports to compare: "))
        sports_means = []
        for sport in range(num_sports):
            selection = input("Enter a sport for comparison: ")
            selection = selection.lower()
            if(check_sport_validity(selection) == False):
                print("Please enter a valid sport.")
                print("")
                continue
            sports_means.append(selection)
        compare_sports_means(sports_means)

    elif option == '3':
        sport1 = input("Enter the first sport for comparison: ")
        sport1 = sport1.lower()
        if(check_sport_validity(sport1) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        sport2 = input("Enter the second sport for comparison: ")
        sport2 = sport2.lower()
        if(check_sport_validity(sport2) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        sports_correlation(sport1, sport2)

    elif option == '4':
        break

    else:
        print("Please select a valid option.")
        print("")
        continue
print("Thank you for using the Chomp Chain Reaction Data Analyzer!")