import pandas as pd
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
    #    return vb_df

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
        #parse through string
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
        sport2_df = retrieve_data_frame(sport2)
        #find start years
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