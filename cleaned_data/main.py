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
    #converts 2 digit years into 4 digit years
    if len(start) == 4:
        return int(start)

    start_num = int(start)
    if start_num <= 26:
        return 2000 + start_num
    else:
        return 1900 + start_num


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

#function for option 2 graph
def plot_sports_records(list_of_sports):
    print("")
    #make option 2 graph

#not done
def sports_correlation(sport1, sport2):
    """Computes the correlation coefficient between two UF sports' win percentages.
    Produces:
        - Scatter plot (sport 1 win% vs. sport 2 win%)
        - Best fit regression line
        """
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
    win1 = list(sport1_df["win%"])
    win2 = list(sport2_df["win%"])

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

#option 4 graph not done
def compare_sports_means(sport1_name, sport2_name):
    sport1_df = retrieve_data_frame(sport1_name)
    sport2_df = retrieve_data_frame(sport2_name)

    # Convert year strings to integer starting years
    sport1_df["start_year"] = [convert_year_to_integer(season) for season in sport1_df["year"]]
    sport2_df["start_year"] = [convert_year_to_integer(season) for season in sport2_df["year"]]

    # Keep original "06-07" labels
    sport1_df["season_label"] = sport1_df["year"]
    sport2_df["season_label"] = sport2_df["year"]

    #user selects interval to analyze over
    start_year_input = int(input("Start Year: "))
    end_year_input = int(input("End Year: "))

    # Filters dfs to selected interval
    sport1_interval_df = sport1_df[(sport1_df["start_year"] >= start_year_input) & (sport1_df["start_year"] <= end_year_input)]
    sport2_interval_df = sport2_df[(sport2_df["start_year"] >= start_year_input) & (sport2_df["start_year"] <= end_year_input)]

    # Determine overlapping seasons
    shared_years = sorted(set(sport1_interval_df["start_year"]) & set(sport2_interval_df["start_year"]))

    if not shared_years:
        print("There are no overlapping seasons between these sports in this interval.")
        return

    #ensures only shared years are being evaluated
    sport1_shared_df = sport1_interval_df[sport1_interval_df["start_year"].isin(shared_years)]
    sport2_shared_df = sport2_interval_df[sport2_interval_df["start_year"].isin(shared_years)]

    # Sort for safety
    sport1_shared_df = sport1_shared_df.sort_values("start_year")
    sport2_shared_df = sport2_shared_df.sort_values("start_year")

    #basic info needed for plotting (years, win%s, labels)
    sport1_years = list(sport1_shared_df["start_year"])
    sport2_years = list(sport2_shared_df["start_year"])

    sport1_win = list(sport1_shared_df["win%"])
    sport2_win = list(sport2_shared_df["win%"])

    sport1_labels = list(sport1_shared_df["season_label"])
    sport2_labels = list(sport2_shared_df["season_label"])

    #overall aroc over given time period
    interval_length = sport1_years[-1] - sport1_years[0]
    sport1_overall_aroc = (sport1_win[-1] - sport1_win[0]) / interval_length
    sport2_overall_aroc = (sport2_win[-1] - sport2_win[0]) / interval_length


    #plot graph
    plt.figure(figsize=(10, 10))
    # Win% performance lines
    plt.plot(sport1_years, sport1_win, linewidth=3, marker="o", color=get_color(sport1_name), label=f"{sport1_name.capitalize()} Win%")

    plt.plot(sport2_years, sport2_win, linewidth=3, marker="o", color=get_color(sport2_name), label=f"{sport2_name.capitalize()} Win%")

    # Shared x-tick labels
    plt.xticks(sport1_years, sport1_labels)

    plt.title(f"Performance Trend + Overall AROC: {sport1_name.capitalize()} vs {sport2_name.capitalize()}\n ({start_year_input}–{end_year_input})")
    plt.xlabel("Season")
    plt.ylabel("Win Percentage")
    plt.legend()
    plt.show()

    #analysis of aroc
    def print_summary(team, start_label, end_label, start_win, end_win):
        total_change = end_win - start_win
        direction = "increased" if total_change > 0 else "decreased" if total_change < 0 else "stayed the same"
        print(f"{team.capitalize()}'s win percentage {direction} by {abs(total_change):.2f} points from {start_label} to {end_label}, averaging {(total_change / interval_length):.2f} points per year.")

    print_summary(sport1_name, sport1_shared_df.iloc[0]["season_label"], sport1_shared_df.iloc[-1]["season_label"], sport1_win[0], sport1_win[-1])
    print_summary(sport2_name, sport2_shared_df.iloc[0]["season_label"], sport2_shared_df.iloc[-1]["season_label"], sport2_win[0], sport2_win[-1])

    print("")
    #make graphs and compare mean values, option 4

running = True
while (running):
    print("1. Compare all sports with the championship season of one sport")
    print("2. Compare specific sports records")
    print("3. Find the correlation between two sports")
    print("4. Compare AROC of records over a specific period")
    print("5. Exit")
    option = input("Type in a number to select an option:")

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
        sports_list = []

        for i in range(num_sports):
            sport = input("Enter the sport for comparison: ")
            sport = sport.lower()
            if (check_sport_validity(sport) == False):
                print("Please enter a valid sport.")
                print("")
                continue
            sports_list.append(sport)
        plot_sports_records(sports_list)

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
        #print correlation coefficient
        #make specific observation???

    elif (option == '4'):
        sport1 = input("Enter your first sport: ")
        sport1 = sport1.lower()
        if  (check_sport_validity(sport1) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        sport2 = input("Enter your second sport: ")
        sport2 = sport2.lower()
        if (check_sport_validity(sport2) == False):
            print("Please enter a valid sport.")
            print("")
            continue
        compare_sports_means(sport1, sport2)

        print("4.")
        #just added the print statement so there wouldn't be an error
        #idk if we wanted to be able to compare just two sports or a bunch so I figured I'd leave input up to you
        #same with input about year range

    elif (option == '5'):
        break

    else:
        print("Please select a valid option.")
        print("")
        continue
print("Thank you for using the Chomp Chain Reaction Data Analyzer!")