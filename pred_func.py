import pandas as pd
import numpy as np

#Reads full dataframe and creates a dataframe with team as index and number of the actual wins for the full season
#used for finding and diminishing margin of error in model
def get_final_standings(df):
    fullteamdf = pd.DataFrame(columns = ['actual total'])
    #loop sums up combination of 1) games where team was hometeam and home team won and 2) games where team was visitteam and visit team won
    for team in df['home team'].unique():
        numwins = np.sum(df.where(team == df['home team'])['hwon']) + np.sum(df.where(team == df['visit team'])['vwon'])
        fullteamdf.loc[team] = [int(numwins)]
    return fullteamdf

#creates dataframe with team as index and includes wins losses, and relevant statistics for prediction to use
#some of the lines get long in this one, and im aware of that, i apoligize for the readability
def create_teamdf(df):
    teamdf = pd.DataFrame(columns = ['wins', 'losses', 'extra_dif', '1run_dif', '2run_dif', 'unearned runs']) #'games back', ])

    #loop uses similar strategy to above function of finding games where team was home and then seeing if hometeam did X, then doing same for visiting team
    for team in df['home team'].unique():
        #for wins, losses and number of games
        numwins = np.sum(df.where(team == df['home team'])['hwon']) + np.sum(df.where(team == df['visit team'])['vwon'])
        numgames = np.sum(df.where(team == df['home team'])['game_counter']) + np.sum(df.where(team == df['visit team'])['game_counter'])
        numlosses = numgames - numwins
        #for difference between wins and losses in extra innings games
        extradf = df.where(df['game outs'] > 54)
        ei_wins = np.sum(extradf.where((team == df['home team']) & (df['hwon'] == 1))['game_counter']) + np.sum(extradf.where((team == df['visit team']) & (df['vwon'] == 1))['game_counter'])
        ei_games = np.sum(extradf.where((team == df['home team']) | (team == df['visit team']))['game_counter'])
        ei_dif = int(2 * ei_wins - ei_games)
        #for difference between wins and losses in 1 run games
        #created ninedf in order to not doublecount extra inning games decided by 1 or 2 runs
        ninedf = df.where(df['game outs'] <= 54)
        onerun_w = np.sum(ninedf.where((team == ninedf['home team']) & (ninedf['margin'] == 1))['game_counter']) + np.sum(ninedf.where((team == ninedf['visit team']) & (ninedf['margin'] == -1))['game_counter'])
        onerun_l = np.sum(ninedf.where((team == ninedf['home team']) & (ninedf['margin'] == -1))['game_counter']) + np.sum(ninedf.where((team == ninedf['visit team']) & (ninedf['margin'] == 1))['game_counter'])
        one_run_dif = int(onerun_w - onerun_l)
        #for difference between wins and losses in 2 run games
        tworun_w = np.sum(ninedf.where((team == ninedf['home team']) & (ninedf['margin'] == 2))['game_counter']) + np.sum(ninedf.where((team == ninedf['visit team']) & (ninedf['margin'] == -2))['game_counter'])
        tworun_l = np.sum(ninedf.where((team == ninedf['home team']) & (ninedf['margin'] == -2))['game_counter']) + np.sum(ninedf.where((team == ninedf['visit team']) & (ninedf['margin'] == 2))['game_counter'])
        two_run_dif = int(tworun_w - tworun_l)

        #Below for calculating in game unearned runs, seems too loosely coordinated to be helpful for predicition
        unearned = np.sum(df.where(team == df['home team'])['hUR']) + np.sum(df.where(team == df['visit team'])['vUR'])
        #combines all new data into teamdf
        teamdf.loc[team] = [int(numwins), int(numlosses), ei_dif, one_run_dif, two_run_dif, int(unearned)]
    #useful data for later prediction
    teamdf['num_games'] = teamdf['wins'] + teamdf['losses']
    teamdf['win_per'] = teamdf['wins'] / teamdf['num_games']

    # #loop used to include 7/4 winning percentage for each team in the original DataFrame, makes later calculations faster
    # for i in range(len(df)):
    #     df.loc[i, 'Hwin%'] = teamdf.loc[df.loc[i, 'home team'], 'win_per']
    #     df.loc[i, 'Vwin%'] = teamdf.loc[df.loc[i, 'visit team'], 'win_per']

    return teamdf

def set_wp(teamdf, df):
    #loop used to include 7/4 winning percentage for each team in the original DataFrame, makes later calculations faster
    df1 = df.copy()
    for i in range(len(df1)):
        hteam = df1.loc[i, 'home team']
        vteam = df1.loc[i, 'visit team']
        df1.loc[i, 'Hwin%'] = teamdf.loc[hteam, 'win_per']
        df1.loc[i, 'Vwin%'] = teamdf.loc[vteam, 'win_per']
    return df1

def calc_sos(teamdf, predf, postdf):
    #Following loop calculates average opponent win percentage and adds it to teamdf
    #I couldnt figure out the smoother way of how to locate the index of teamdf and compare it to df['home team'], but this got the job done
    #wouldve probably done more research to figure out if there is a better way but the dataframe is only 30 rows so no ones gonna arrest me
    teamdf = teamdf.reset_index()
    for i in range(len(teamdf)):
        teamdf.loc[i, 'opp_win_per'] = np.sum(predf.where(teamdf.loc[i, 'index'] == predf['home team'])['Vwin%']) + np.sum(predf.where(teamdf.loc[i, 'index'] == predf['visit team'])['Hwin%'])
        teamdf.loc[i, 'owp_remain'] = np.sum(postdf.where(teamdf.loc[i, 'index'] == postdf['home team'])['Vwin%']) + np.sum(postdf.where(teamdf.loc[i, 'index'] == postdf['visit team'])['Hwin%'])
    teamdf = teamdf.set_index('index')
    teamdf['opp_win_per'] /= teamdf['num_games']
    teamdf['owp_remain'] /= (162 - teamdf['num_games'])
    teamdf['owp_diff'] = teamdf['opp_win_per'] - teamdf['owp_remain']
    return teamdf

#adds more columns to teamdf, mostly results of different prediction methods
def predict(teamdf, fulldf):
    #win adjust is the number of wins that will be added to a teams current wins as a result fo model
    teamdf['win_adjust'] = -(.5*teamdf['extra_dif'] + .33*teamdf['1run_dif']  + .25*teamdf['2run_dif'])
    teamdf['win_adjust'] += (100 * teamdf['owp_diff'])
    #teamdf = teamdf.sort_values(by=['win_adjust'])
    #extrap total is the number of wins a team would finish with at current win percentage
    teamdf['extrap_total'] = 162 * teamdf['win_per']
    #model creates winning percentage that team should've had and then extrapolates that for remaining games
    teamdf['proj_total'] = (162 - teamdf['num_games']) * (teamdf['wins'] + teamdf['win_adjust']) / (teamdf['num_games'])
    teamdf['proj_total'] += teamdf['wins']
    #500 total is number of wins team would have if they won half of their remaining games
    teamdf['500_extrap_total'] = (162 - (teamdf['num_games'])) * .5 + teamdf['wins']
    #megres actual win total in
    teamdf = teamdf.merge(fulldf, how='outer', left_index=True, right_index=True)
    #calculates difference from actual total for each
    teamdf['error'] = teamdf['proj_total'] - teamdf['actual total']
    teamdf['extrap_error'] = teamdf['extrap_total'] - teamdf['actual total']
    teamdf['500_extrap_error'] = teamdf['500_extrap_total'] - teamdf['actual total']

    teamdf = teamdf[['error', 'extrap_error', '500_extrap_error']]
    teamdf = teamdf.sort_values(by=['error'])
    print(teamdf)#REMOVE

    return np.mean(abs(teamdf['error'])), np.mean(abs(teamdf['extrap_error'])), np.mean(abs(teamdf['500_extrap_error']))
