import pandas as pd
import numpy as np
import pred_func as f
import os.path

homedir = os.path.expanduser("~") #to open the file from another computer
df = pd.read_csv(homedir + '/Desktop/playoff_predicter/retrosheet_data1/retrosheet0918.csv', low_memory=False)

#narrows down file to relevant stats
df = df[['Date','home team', 'visit team', 'home score', 'visit score', 'game outs', 'htER', 'vtER']]

#adds columns that make future operations easier
df['game_counter'] = 1
df['margin'] = df['home score'] - df['visit score'] #note margin is home - away
df['hwon'] = 0
df.loc[df['margin'] > 0, 'hwon'] = 1
df['vwon'] = abs(df['hwon'] - 1)
df['vUR'] = df['visit score'] - df['htER']
df['hUR'] = df['home score'] - df['vtER']

#ensure that final error numbers are accurate
sum_error = 0
sum_extrap_error = 0
sum_half_error = 0

list = []

for year in range(2009,2018):
    #using copy so dataframe can be narrowed down each time, significantly increasing speed
    dfcopy = df.copy()
    dfcopy = dfcopy[(dfcopy['Date'] > (year * 10000)) & (dfcopy['Date'] < (year * 10001))]
    #LOOK UP DOCUMENTATION FOR RESET REMOVE OLD INDEX
    dfcopy = dfcopy.reset_index()

    season_df = f.get_final_standings(df=dfcopy)
    #narrows down dataframes to before and after july fourth
    pre_julydf = dfcopy[dfcopy['Date'] < (year * 10000 + 704)]
    post_julydf = dfcopy[dfcopy['Date'] >= (year * 10000 + 704)]
    post_julydf = post_julydf.reset_index()

    df1 = f.create_teamdf(df=pre_julydf)
    pre_julydf = f.set_wp(teamdf=df1, df=pre_julydf)
    post_julydf = f.set_wp(teamdf=df1, df=post_julydf)
    df1 = f.calc_sos(teamdf=df1, predf=pre_julydf, postdf=post_julydf)
    error = f.predict(teamdf=df1, fulldf=season_df)

    #output the error by season for all three methods
    print('Average margin of error in ',year, ': ', error[0])
    print('Average margin of error (continuing current percentage): ', error[1])
    print('Average margin of error (winning half of remaining games): ', error[2], '\n')

    #build sum of all errors to find mean after loop
    sum_error += error[0]
    sum_extrap_error += error[1]
    sum_half_error += error[2]

    list.append(error[1] - error[0])

#finds mean over 10 seasons and outputs
mean_error = sum_error/10
mean_extrap_error = sum_extrap_error/10
mean_half_error = sum_half_error/10
print('Average total margin of error: ', mean_error)
print('Average total margin of error (continuing current percentage): ', mean_extrap_error)
print('Average total margin of error (winning half of remaining games): ', mean_half_error)

print('\nMean difference in error of predictions (model - extrapolated win %): ', np.mean(list))
print('standard deviation: ', np.std(list))
print(list)

     # The information used here was obtained free of
     # charge from and is copyrighted by Retrosheet.  Interested
     # parties may contact Retrosheet at 20 Sunset Rd.,
     # Newark, DE 19711.
