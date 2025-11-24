import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sb_df = pd.read_csv('softball.csv')
bb_df = pd.read_csv('baseball.csv')
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

def find_start_year(sport_list):
    return True

def retrieve_data_frame(sport):
    if sport == 'softball':
        return sb_df
    #elif sport == 'baseball':
    #   return bb_df
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
    #    return vb_d


def convert_year_to_integer(year_string):
    return int(year_string.split("-")[0])

#function for option 2 graph
def plot_sports_records(list_of_sports):
    #make option 2 graph

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
        #print('2')

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
        #sport1_df = retrieve_data_frame(sport1)
        #sport2_df = retrieve_data_frame(sport2)
        #find start years
        #make graph
        #print correlation coefficient
        #make specific observation???

    elif (option == '4'):
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
        compare_sports_means(sports_list)
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