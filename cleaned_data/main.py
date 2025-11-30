import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sb_df = pd.read_csv('softball.csv')
bb_df = pd.read_csv('baseball.csv')
#test comment
#uncomment when CSVs are available
fb_df = pd.read_csv('football.csv')
#gym_df = pd.read_csv('gymnastics.csv')
#ten_df = pd.read_csv('tennis.csv')
#wten_df = pd.read_csv('womenstennis.csv')
bk_df = pd.read_csv('basketball.csv')
#wbk_df = pd.read_csv('womensbasketball.csv')
#soc_df = pd.read_csv('soccer.csv')
#vb_df = pd.read_csv('volleyball.csv')

def check_sport_validity(sport):
    if (sport!= "football" and sport!= "basketball" and sport!= "baseball" and sport!= "softball" and sport!= "women's basketball" and sport!= "tennis" and sport!= "women's tennis" and sport!= "soccer" and sport!= "gymnastics" and sport!= "volleyball"):
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
    #elif sport == 'gymnastics':
    #   return gym_df
    #elif sport == 'tennis':
    #   return ten_df
    #elif sport == "women's tennis":
    #   return wten_df
    elif sport == 'basketball':
        return bk_df
    #elif sport == "women's basketball":
    #    return wbk_df
    #elif sport == 'soccer':
    #    return soc_df
    #elif sport == "volleyball":
    #    return vb_df
def convert_year_to_integer(year_string):
    start = year_string.split("-")[0]
    return int(start)


def find_start_year(sport_list):
    sport_df = retrieve_data_frame(sport_list[0])
    start_year = int(sport_df['year'].iloc[-1][0:2])
    for i in range (len(sport_list)):
        sport_df = retrieve_data_frame(sport_list[i])
        if (int(sport_df['year'].iloc[-1][0:2]) < start_year):
            start_year = int(sport_df.iloc[-1,0][0:2])
    start_year_str = f'{start_year}-{start_year+1}'
    return start_year_str


def get_color(sport):
    if sport == 'softball':
        return 'olive'
    elif sport == 'baseball':
        return 'green'
    elif sport == 'football':
        return 'orange'
    elif sport == 'gymnastics':
        return 'purple'
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
        return 'brown'

#not done
def first_graph():
    #complete_sport_list = ['football', 'basketball','softball','baseball','gymnastics','volleyball','tennis',"women's tennis",'soccer',"women's basketball"]
    #start_year = find_start_year(complete_sport_list)
    temp_sport_list = ['softball','basketball','baseball','football']
    for i in range (len(temp_sport_list)):
        sport_df = retrieve_data_frame(temp_sport_list[i])
        sport_df.plot(x ='year',y = 'win%')
    plt.title(f'Win/Loss Records of Florida Gators Sports from 1906 to Present')
    plt.xlabel('Season', fontsize = 18)
    plt.ylabel('Record Percentage', fontsize = 18)
    plt.show()


#check if it stacks
def segmented_bar_chart():
    temp_sport_list = ['softball','basketball','baseball','football']
    x = (retrieve_data_frame('baseball'))['year'].tolist()
    x_vals = []
    for i in range (1,len(retrieve_data_frame('football')['year'].tolist())+1):
        x_vals.append(retrieve_data_frame('football')['year'].iloc[-i])
    start_year_int = int(find_start_year(temp_sport_list)[0:2])
    for i in range(len(temp_sport_list)):
        color = get_color(temp_sport_list[i])
        y = []
        sport_df = retrieve_data_frame(temp_sport_list[i])
        if (convert_year_to_integer(sport_df['year'].iloc[-1]) > start_year_int):
            for k in range(int(sport_df['year'].iloc[-1][0:2]) - start_year_int - 1):
                y.append(0)
        for j in range(1,len(sport_df['year'].tolist())+1):
            if ((sport_df['national_championship'].iloc[-j]) == 'yes'):
                y.append(1)
            else:
                y.append(0)
        plt.bar(x_vals, y, color=color, label = temp_sport_list[i])
    plt.xlabel('Season', fontsize=10)
    plt.ylabel('Number of Championship Wins', fontsize=10)
    #plt.yticks([1.0],[1])
    plt.title('Champsionship Wins per Year')
    #plt.xticks(['26-27','30-31','40-41','50-51','60-61','70-71','80-81','90-91','00-01','10-11','20-21'],['26-27','30-31','40-41','50-51','60-61','70-71','80-81','90-91','00-01','10-11','20-21'],fontsize = 8)
    plt.legend()
    plt.show()

segmented_bar_chart()

#option 1 graph function,
def champ_sports_comparison(champ_sport):
    pass

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
            strenth = "moderate"
        elif abs(r) > 0 and abs(r) < .3:
            strength = "weak"
        elif abs(r) == 0:
            strength = "none"

        print(f"With a correlation coefficient of {r:.2f}, the win percentages over time of {sport1.capitalize()} and {sport2.capitalize()} have a {strength}, {direction} correlation.")


#ensure it works when csvs are correct, option 2 function
def compare_sports_means(sport_list):
    #loads all df into a dictionary
    sports_df = {}
    for sport in sport_list:
        df = retrieve_data_frame(sport)
        for season in df["year"]:
            df["start_year"] = [convert_year_to_integer(season)]
            df["season_label"] = df["year"]
            sports_df[sport] = df

    #user selects interval to analyze over
    start_year_input = int(input("Start Year: "))
    end_year_input = int(input("End Year: "))

    # Filters dfs to selected interval
    sport_interval_dfs = {}
    for sport, df in sports_df.items():
        interval_df = df[(df["start_year"] >= start_year_input) & (df["start_year"] <= end_year_input)]
        sport_interval_dfs[sport] = interval_df

    # Determine overlapping seasons
    for df in sport_interval_dfs.values():
        year_sets = set(df["start_year"])
    shared_years = sorted(set.intersection(*year_sets))

    if not shared_years:
        print("There are no overlapping seasons between these sports in this interval.")
        return

    #ensures only shared years are being evaluated
    for sport, df in sports_df.items():
        filtered_df[sport] = df[df["start_year"].isin(shared_years)]
        sorted_df = filtered_df.sort_values("start_year")
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

        print(f"{sport.capitalize()}'s win percentage {direction} by {abs(total_change):.2f} points from {labels[0]} to {labels[-1]}.")
        print(f"{sport.capitalize()}'s win percentage {direction} by {abs(average_change):.2f} on average from year to year.")

    #labels for x-axis
    example_sport = sport_list[0]
    plt.xticks(interval_df[example_sport]["start_year"], interval_df[example_sport]["season_label"])

    #title
    for sport in sport_list:
        sports_names_formatted = ", ".join(sport.capitalize())
    year_range = f"{start_year_input}-{end_year_input}"
    full_title = f"Performance Trends: {sports_names_formatted}\n {year_range}"
    plt.title(full_title)

    plt.xlabel("Season")
    plt.ylabel("Win Percentage")
    plt.legend()
    plt.show()


running = True
while (running):
    print("1. Compare all sports with the championship season of one sport")
    print("2. Compare specific sports records")
    print("3. Find the correlation between two sports")
    print("4. Compare AROC of records over a specific period")
    print("5. Exit")
    option = input("Type in a number to select an option: ")

    if (option == '1'):
        champ_sport = input("Enter the sport for comparison: ")
        champ_sport = champ_sport.lower()
        if (check_sport_validity(champ_sport) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        champ_df = retrieve_data_frame(champ_sport)
        champ_color = get_color(champ_sport)
        #make graphs

    elif (option == '2'):
        num_sports = int(input("Enter the number of sports to compare: "))
        sports_means = []
        for sport in range(num_sports):
            selection = input("Enter the sport for comparison: ")
            selection = selection.lower()
            if check_sport_validity(selection) == False:
                print("Please enter a valid sport.")
                print("")
                continue
            sports_means.append(selection)
        compare_sports_means(sports_means)


    elif (option == '3'):
        sport1 = input("Enter your first sport: ")
        sport1 = sport1.lower()
        if(check_sport_validity(sport1) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        sport2 = input("Enter your second sport: ")
        sport2 = sport2.lower()
        if(check_sport_validity(sport2) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        sports_correlation(sport1, sport2)

    elif (option == '5'):
        break

    else:
        print("Please select a valid option.")
        print("")
        continue
print("Thank you for using the Chomp Chain Reaction Data Analyzer!")