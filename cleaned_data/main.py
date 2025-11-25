import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sb_df = pd.read_csv('softball.csv')
bb_df = pd.read_csv('baseball.csv')
#test comment
#uncomment when CSVs are available
#fb_df = pd.read_csv('football.csv')
#bb_df = pd.read_csv('baseball.csv')
#gym_df = pd.read_csv('gymnastics.csv')
#ten_df = pd.read_csv('tennis.csv')
#wten_df = pd.read_csv('womenstennis.csv')
#bk_df = pd.read_csv('basketball.csv')
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
    #elif sport == 'football':
    #   return fb_df
    #elif sport == 'gymnastics':
    #   return gym_df
    #elif sport == 'tennis':
    #   return ten_df
    #elif sport == "women's tennis":
    #   return wten_df
    #elif sport == 'basketball':
    #    return bk_df
    #elif sport == "women's basketball":
    #    return wbk_df
    #elif sport == 'soccer':
    #    return soc_df
    #elif sport == "volleyball":
    #    return vb_df

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
    temp_sport_list = ['softball']
    for i in range (len(temp_sport_list)):
        sport_df = retrieve_data_frame(temp_sport_list[i])
        sport_df.plot(x ='year',y = 'win%')
    plt.title(f'Win/Loss Records of Florida Gators Sports from 1906 to Present')
    plt.xlabel('Season', fontsize = 18)
    plt.ylabel('Record Percentage', fontsize = 18)
    plt.show()

def convert_year_to_integer(year_string):
    return int(year_string.split("-")[0])

#function for option 2 graph
def plot_sports_records(list_of_sports):
    print("")
    #make option 2 graph

#check if it stacks
def segmented_bar_chart():
    temp_sport_list = ['softball', 'baseball']
    x = (retrieve_data_frame('baseball'))['year'].tolist()
    x_vals = []
    for i in range (1,len(retrieve_data_frame('baseball')['year'].tolist())+1):
        x_vals.append(retrieve_data_frame('baseball')['year'].iloc[-i])
    start_year_int = int(find_start_year(temp_sport_list)[0:2])
    for i in range(len(temp_sport_list)):
        color = get_color(temp_sport_list[i])
        y = []
        sport_df = retrieve_data_frame(temp_sport_list[i])
        if (convert_year_to_integer(sport_df['year'].iloc[-1]) > start_year_int):
            for i in range(int(sport_df['year'].iloc[-1][0:2]) - start_year_int - 1,):
                y.append(0)
        for j in range(1,len(sport_df['year'].tolist())+1):
            if ((sport_df['national_championship'].iloc[-j]) == 'yes'):
                y.append(1)
            else:
                y.append(0)
        plt.bar(x_vals, y, color=color)
    plt.xlabel('Season', fontsize=10)
    plt.ylabel('Number of Championship Wins', fontsize=10)
    plt.yticks([1.0],[1])
    plt.xticks(['26-27','30-31','40-41','50-51','60-61','70-71','80-81','90-91','00-01','10-11','20-21'],['26-27','30-31','40-41','50-51','60-61','70-71','80-81','90-91','00-01','10-11','20-21'],fontsize = 8)
    plt.show()
segmented_bar_chart()

def sports_correlation(sport1, sport2):
    """Computes the correlation coefficient between two UF sports' win percentages.
    Produces:
        - Scatter plot (sport 1 win% vs. sport 2 win%)
        - Best fit regression line
        """
    sport1_df = retrieve_data_frame(sport1)
    sport2_df = retrieve_data_frame(sport2)

    #Converts csv years to integers to work with in the graph
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

    #determines the shared start year between the two sports, and considers everything that year and beyond
    shared_start_year = find_start_year([sport1_df, sport2_df])
    sport1_df = sport1_df[sport1_df["integer_years"] >= shared_start_year]
    sport2_df = sport2_df[sport2_df["integer_years"] >= shared_start_year]

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


def compare_sports_means(list_of_sports):
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
        #user input needed
        print('2')

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
        sport1_df = retrieve_data_frame(sport1)
        sport1_color = get_color(sport1)
        sport2_df = retrieve_data_frame(sport2)
        sport2_color = get_color(sport2)
        start_year = find_start_year([sport1,sport2])
        #make graph
        #print correlation coefficient
        #make specific observation???

    elif (option == '4'):
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