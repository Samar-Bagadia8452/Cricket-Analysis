#!/usr/bin/env python
# coding: utf-8

# # Import the Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# # Loading the Datasets

# # Deliveries dataset

# In[2]:


df = pd.read_csv(r'C:\Users\helLO\Downloads\archive (2)\deliveries.csv')
df.head()


# # Matches dataset

# In[3]:


df1 = pd.read_csv(r'C:\Users\helLO\Downloads\archive (2)\matches.csv')
df1.head()


# # Data Cleaning Part

# ***There have been multiple names for the Same Team- the Data is cleaned in the following portion.***

# In[4]:


df1.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
df1.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
df1.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
df1.venue.replace({'Feroz Shah Kotla Ground':'Feroz Shah Kotla',
                    'M Chinnaswamy Stadium':'M. Chinnaswamy Stadium',
                    'MA Chidambaram Stadium, Chepauk':'M.A. Chidambaram Stadium',
                     'M. A. Chidambaram Stadium':'M.A. Chidambaram Stadium',
                     'Punjab Cricket Association IS Bindra Stadium, Mohali':'Punjab Cricket Association Stadium',
                     'Punjab Cricket Association Stadium, Mohali':'Punjab Cricket Association Stadium',
                     'IS Bindra Stadium':'Punjab Cricket Association Stadium',
                    'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi International Stadium',
                    'Rajiv Gandhi Intl. Cricket Stadium':'Rajiv Gandhi International Stadium'},regex=True,inplace=True)


# In[5]:


df.shape


# In[6]:


df1.shape


# In[7]:


df.info()


# In[8]:


df1.info()


# In[9]:


df.describe()


# In[10]:


df1.describe()


# In[11]:


df.columns


# In[12]:


df1.columns


# In[13]:


df.isnull().count()


# In[14]:


df1.isnull().count()


# In[15]:


df.isnull().sum()


# In[16]:


df1.isnull().sum()


# In[164]:


df1['season'].value_counts()


# In[165]:


df1['city'].value_counts()


# In[166]:


df1['result'].unique()


# In[169]:


df1['dl_applied'].unique()


# In[170]:


df1['team1'].value_counts()


# In[171]:


df1['winner'].value_counts()


# In[ ]:





# # Data Visualization 

# # List of Top 10 Cities where most number of matches have been played

# In[17]:


city_counts = df1.groupby('city')['city'].count().reset_index(name='Match Counts')

top_cities_order = city_counts.sort_values(by='Match Counts', ascending=False)

top_cities = top_cities_order[:10]

print('Top 10 cities with the maximum number of Matches Played:\n', top_cities)

plt.figure(figsize=(9,9))
plt.pie(top_cities['Match Counts'], labels=top_cities['city'], autopct='%1.1f%%', startangle=30)
plt.axis('equal')
plt.title('Top Cities that have hosted IPL Matches', size=20)
plt.show()


# # List of Top 10 venues where most number of matches have been played

# In[18]:


venue_counts = df1.groupby('venue')['venue'].count().reset_index(name='Match Counts')

top_venues_order = venue_counts.sort_values(by='Match Counts', ascending=False)

top_venues = top_venues_order[:10]

print('Top 10 Stadiums with the maximum number of Matches Played:\n', top_venues)

plt.figure(figsize=(9,9))
plt.pie(top_venues['Match Counts'], labels=top_venues['venue'], autopct='%1.1f%%', startangle=40)
plt.axis('equal')
plt.title('Top Stadiums that have hosted IPL Matches', size=20)
plt.show()


# # List of Top 10 highest scoring Batsman

# In[19]:


batting_tot = df.groupby('batsman')['batsman_runs'].sum().reset_index(name='Runs')

batting_sorted = batting_tot.sort_values(by='Runs', ascending=False)

top_batsman = batting_sorted[:10]

print('The Top 10 Batsmen in the Tournament are:\n', top_batsman)

fig = px.bar(top_batsman, x='batsman', y='Runs', hover_data=['batsman'], color='Runs', title='Top 10 Batsmen in IPL - Seasons 2008-2019')
fig.show()


# ***Here we can see that Virat Kholi & Suresh Raina are the high scoring batsman in the IPL history***

# # List of Top 10 highest scorers in a match 

# In[20]:


batting_ings = df.groupby(['match_id', 'batsman'])['batsman_runs'].sum().reset_index(name='Innings Runs')

batting_ings_sorted = batting_ings.sort_values(by='Innings Runs', ascending=False)

top_batsman_scores = batting_ings_sorted[:10]

batsman_ball_faced = df.groupby(['match_id', 'batsman'])['batsman_runs'].count().reset_index(name='Balls Faced')

batsman_performance = pd.merge(top_batsman_scores, batsman_ball_faced, how='inner', on=['match_id', 'batsman'])

batsman_performance['Strike Rate for Match'] = batsman_performance['Innings Runs'] * 100 / batsman_performance['Balls Faced']

batsman_innings = pd.merge(batsman_performance, df, how='inner', on=['match_id', 'batsman'])

batsman_innings_req = batsman_innings.iloc[:, 1:8]

batsman_innings_req_2 = batsman_innings_req.drop_duplicates()

print('The Top 10 Batting Performances in IPL History are:\n', batsman_innings_req_2)

x = batsman_innings_req_2['batsman']
y1 = batsman_innings_req_2['Strike Rate for Match']
y2 = batsman_innings_req_2['Innings Runs']

plt.figure(figsize=(10, 5))
plt.scatter(x, y1)
plt.scatter(x, y2)
plt.xlabel('Batsman', size=15)
plt.ylabel('Strike Rate/Innings Score', size=15)
plt.title('IPL Best Batting Performances in a Match', size=20)
plt.xticks(rotation=60)
plt.legend(['Strike Rate', 'Runs'], prop={'size':20})
plt.show()


# ***Here we can see that Chris Gayle hits the highest score 175 runs in a match in IPL.***
# ***Chris Gayle and AB de Villiers have appeared twice in the Top 10 run scorer list. Both have represented RCB, when they have enlisted their Top 2 scores- which are in the Top 10 list of IPL History- making RCB achieve Top 4 spots out of 10, Followed by CHennai Super Kings with 2 spots. A very surprising observation lies in the fact that all the bowling teams against whom the top scores in a match is achieved are different- and comprises of 4 Teams who have been IPL champions (Out of 5).***

# # List of Top 10 Bowlers with highest number of wickets

# In[21]:


bowling_wickets = df[df['dismissal_kind'] != 'run out']

bowling_tot = bowling_wickets.groupby('bowler')['dismissal_kind'].count().reset_index(name='Wickets')

bowling_top = bowling_tot.sort_values(by='Wickets', ascending=False)

top_bowlers = bowling_top.iloc[:10, :]

print('The Top Wicket Takers in the Tournament are:\n', top_bowlers)

fig = px.bar(top_bowlers, x='bowler', y='Wickets', hover_data=['bowler'], color='Wickets', title='Top 10 Bowlers in IPL - Seasons 2008-2019')
fig.show()


# ***Lasith Malinga is the highest wicket taker in IPL.***
# ***Run Out is not considered as a wicket in the Bowler's account- so we remove them in  first step***

# # List of Top 10 Wicket Taker in a match of IPL

# In[172]:


match_bowling_tot = bowling_wickets.groupby(['match_id', 'bowler'])['dismissal_kind'].count().reset_index(name='Wickets')

match_bowling_top = match_bowling_tot.sort_values(by='Wickets', ascending=False)

match_top_bowlers = match_bowling_top.iloc[:10]

match_bowling_runs = df.groupby(['match_id', 'bowler'])['total_runs'].sum().reset_index(name='Runs Conceded')

match_bowler_performance = pd.merge(match_top_bowlers, match_bowling_runs, on=['match_id', 'bowler'])

match_bowler_performance['Runs per Wicket'] = match_bowler_performance['Runs Conceded'] / match_bowler_performance['Wickets']

bowler_innings = pd.merge(match_bowler_performance, df, on=['match_id', 'bowler'])

bowler_innings_req_2 = bowler_innings.iloc[:, 1:8].drop_duplicates()

print('The Top 10 Bowling Performances in IPL History are:\n', bowler_innings_req_2)

x = bowler_innings_req_2['bowler']
y1 = bowler_innings_req_2['Wickets']
y2 = bowler_innings_req_2['Runs per Wicket']

plt.figure(figsize=(10, 5))
plt.scatter(x, y1)
plt.plot(x, y2, 'r')
plt.xlabel('Bowlers', size=15)
plt.ylabel('Runs per Wicket', size=15)
plt.title('IPL Best Bowling Performances in a Match', size=20)
plt.xticks(rotation=90)
plt.legend(['Runs per Wicket', 'Wickets'], prop={'size':15})
plt.show()


# ***Here we can see that Adam Zampa is the most economic bowler in an IPL Match.***
# ***Run Out is not considered as a wicket in the Bowler's account- so we remove them in  first step.***
# ***The Team whose players have taken most number of 5+ wickets in a match are Mumbai Indians. No wonder, they have been awarded the trophy most number of times, due to their outstanding performances. The Team against which the maximum number of 5 wickets have been recorder is Sunrisers Hyderabad. This is somewhat due to the lack of middle order batsmen in Sunrisers team in the recent years.***

# # List of Top 10 Fielders (including Wicket Keepers)

# In[23]:


fielder_list_counts = df.groupby('fielder')['dismissal_kind'].count().reset_index(name='Dismissals')

fielder_list_max = fielder_list_counts.sort_values(by='Dismissals', ascending=False)

top_fielders = fielder_list_max.iloc[:10, :]

print('The Best Fielders (and WicketKeepers) in the Tournament are:\n', top_fielders)

fig = px.bar(top_fielders, x='fielder', y='Dismissals', hover_data=['fielder'], color='Dismissals', title='Top 10 Fielders in IPL - Seasons 2008-2019')
fig.show()


# ***Here we can see that Mahendra Singh Dhoni the player having most dismissals under his name in IPL.***
# 
# ***The output gives you the list and visualization of the top 10 fielders (including wicketkeepers) in the IPL from 2008 to 2019, based on the number of dismissals they were involved in. "Dismissals" include all instances where the fielder contributed to getting a batsman out, such as catches, stumpings, and run-outs.***
# 
# ***The bar chart provides a visual representation of these top 10 fielders, showing how many dismissals each has made, which gives a clear indication of their fielding performance over the tournament's history. This data is particularly useful for analyzing the most effective fielders and wicketkeepers in the IPL during the specified period.***

# ### Calculating the Strike Rate of a batsman who has scored more than or equal to a Target Run (The Target Run variable can be changed as per requirement- We have consdiered that the batsman has scored a minimum of 1000 runs)

# In[46]:


Target_run = 1000
batting_tot = df.groupby('batsman')['batsman_runs'].sum().reset_index(name='Runs')
batsman_balls_faced_count = df.groupby('batsman').size().reset_index(name='Balls Faced')

batsman_runs_balls = pd.merge(batting_tot, batsman_balls_faced_count, on='batsman', how='outer')

batsman_runs_balls['Strike Rate'] = (batsman_runs_balls['Runs'] / batsman_runs_balls['Balls Faced']) * 100

plt.scatter(batsman_runs_balls['Runs'], batsman_runs_balls['Strike Rate'])
plt.axhline(y=np.mean(batsman_runs_balls['Strike Rate']), color='r', linestyle='--')
plt.xlabel('Batsman Runs', size=15)
plt.ylabel('Strike Rate', size=15)
plt.title('Overall Runs vs Strike Rate Analysis', size=25)
plt.show()


# In[59]:


Target_run = 1000
batsman_strike_rate_list = batsman_runs_balls.sort_values(by='Strike Rate', ascending=False)
batsman_strike_rate_above_target_runs = batsman_strike_rate_list[batsman_strike_rate_list['Runs'] >= Target_run]
top_strike_rate_batsman = batsman_strike_rate_above_target_runs[['batsman', 'Runs', 'Strike Rate']].head(10)

print(f'The Top 10 batsmen having highest strike rate, scoring at least {Target_run} Runs:\n', top_strike_rate_batsman)

plt.bar(top_strike_rate_batsman['batsman'], top_strike_rate_batsman['Strike Rate'], color='r')
plt.scatter(top_strike_rate_batsman['batsman'], top_strike_rate_batsman['Strike Rate'], color='g')
plt.xlabel('Batsman', size=15)
plt.ylabel('Strike Rate', size=15)
plt.title('Top 10 Batsmen Strike Rate Analysis', size=25)
plt.xticks(rotation=60)
plt.show()


# ***Here we can see that Andre Russel - The player with the best strike rate overall.***
# ***The Strike Rate is basically defined as the runs scored by the Batsman if he faces 100 Balls. A Strike Rate of 179.95, achieved by AD Russel, is a huge value in the game of Cricket and hence he is regarded as one of the best hard-hitters and finishers of the game. Also there are many outliers as evident from the scatterplot. B. Stanlake has scored 5 runs from 2 balls- which makes his strike rate as high as 250.00, but that won't be a suitable comparison to the players who have maintained their performance through all the seasons.***

# ### Calculating the Economy rate of Bowlers who have bowled more than the entered Ball Limit (The Ball Limit variable can be changed as per requirement- We have considered the bowler has atleast bowled 1000 deliveries).

# In[57]:


bowling_runs = df.groupby('bowler')['total_runs'].sum().reset_index(name='Runs Conceeded')
bowled_balls = df.groupby('bowler')['ball'].count().reset_index(name='Balls Bowled')

bowler_stats = pd.merge(bowling_runs, bowled_balls, on='bowler', how='outer')

bowler_stats['Economy Rate'] = (bowler_stats['Runs Conceeded'] / bowler_stats['Balls Bowled']) * 6

plt.figure(figsize=(10, 6))
plt.scatter(bowler_stats['Balls Bowled'], bowler_stats['Economy Rate'], color='g')
plt.xlabel('Balls Bowled', size=15)
plt.ylabel('Economy Rate', size=15)
plt.title('Balls vs Economy Rate Analysis', size=25)
plt.show()


# In[58]:


ball_limit = 1000
bowler_best_economy_rate = bowler_stats.sort_values(by='Economy Rate', ascending=True)
bowler_best_economy_rate_condition = bowler_best_economy_rate[bowler_best_economy_rate['Balls Bowled'] >= ball_limit]

top_10_economy=bowler_best_economy_rate_condition.loc[:,['bowler','Balls Bowled','Economy Rate']][0:10]
print('The Top 10 bowlers having best economy rate, bowling atleast {} balls:\n'.format(Ball_Limit),top_10_economy)

plt.plot(top_10_economy['bowler'],top_10_economy['Economy Rate'],color='y')
plt.scatter(top_10_economy['bowler'],top_10_economy['Economy Rate'],color='b')
plt.xlabel('Bowlers',size=15)
plt.ylabel('Economy Rate',size=15)
plt.title('Top 10 Bowler Economy Rate Analysis',size=25)
plt.xticks(rotation=60)


# ***Here we can see that Dale Steyn is the best bowler with the best economy rate in IPL.***
# ***The Scatter Plot depicts there are many existing outliers. Players having very low ball counts have shown very high values of economy rate, as well as extremely low values- The counts of which are low, but there are examples in the dataset. These are the potential outliers, which we need to ignore in our analysis. Hence we have chosen the bowlers who have bowled a particular number of balls,and thus we are able to identify the best choice among the entire list.***

# # List of the Players who has achieved highest number of 'Man of the Match Awards'

# In[78]:


motm = df1['player_of_match'].value_counts().reset_index()
motm.columns = ['player_of_match', 'Man of the Match Awards']

motm_sort = motm.sort_values(by='Man of the Match Awards', ascending=False)
motm_top = motm_sort.head(10)

plt.plot(motm_top['player_of_match'],motm_top['Man of the Match Awards'],color='b')
plt.bar(motm_top['player_of_match'],motm_top['Man of the Match Awards'],color='grey')
plt.xlabel('Players')
plt.ylabel('Man of the Match Award Count')
plt.title('Top 10 Players who have won most the Man of the Match trophies',size=15)
plt.xticks(rotation=90)


# ***Most of the 'Match of the Man' award winners have been Batsmen and All rounders. Bowlers are not present in the Top 10 list.***
# 
# ***This is clearly indicative of the bias against the bowlers in the Tournament. Let me quote an icident in this regard. In the 2009 Season, Anil Kumble had achieved a 5 Wicket haul, with an economy rate of 1.57 in a match for Royal Challengers bangalore- which is still one of the best performnaces in the history of IPL. But he was not awarded a 'man of the Match' award for that Match. It went to Rahul Dravid for scoring a 48*.***

# ### Best all rounder performance- Considering Batting Factor, Bowling Factor and Fielding Factor.These parameters can be manipulated as per requirement. Showing the Top 10 list. 

# In[88]:


batting_factor=0.5
bowling_factor=15.0
fielding_factor=10.0


all_rounding_1 = pd.merge(batting_sorted, bowling_top, left_on='batsman', right_on='bowler', how='inner')
all_rounding_2 = pd.merge(all_rounding_1, fielder_list_max, left_on='batsman', right_on='fielder', how='left')

# Calculate the Overall Score directly
all_rounding_2['Overall Score'] = (all_rounding_2['Runs'] * batting_factor +
                                   all_rounding_2['Wickets'] * bowling_factor +
                                   all_rounding_2['Dismissals'] * fielding_factor)

# Group by batsman and aggregate the performance
all_rounding_performance = all_rounding_2.groupby(['batsman', 'Runs', 'Wickets', 'Dismissals'])['Overall Score'].sum().reset_index()

# Sort and get the top 10 performers
best_all_round_performance = all_rounding_performance.sort_values(by='Overall Score', ascending=False)
best_overall = best_all_round_performance.head(10)

print('The top 10 best players overall are:\n', best_overall)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(best_overall['batsman'], best_overall['Runs'] * batting_factor, 'g', marker='o', linestyle='-', label='Run Points')
plt.plot(best_overall['batsman'], best_overall['Wickets'] * bowling_factor, 'r', marker='o', linestyle='-', label='Wicket Points')
plt.plot(best_overall['batsman'], best_overall['Dismissals'] * fielding_factor, 'y', marker='o', linestyle='-', label='Dismissal Points')
plt.plot(best_overall['batsman'], best_overall['Overall Score'], 'b', marker='o', linestyle='-', label='Overall Score')
plt.xlabel('The Top 10 Performers', size=15)
plt.ylabel('Scoring Units', size=15)
plt.xticks(rotation=90)
plt.title('Overall Performance by Top 10 Performers in IPL-2008-2019', size=20)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ***PS. We have calculated the Overall Scores based on the following metrics:***
# 
# ***Batting Factor=0.5, i.e. For each Run scored, the overall contribution of the player will be 0.5 ,***
# ***Bowling Factor=15,i.e. For each Wicket Taken, the overall contribution of the player will be 15 ,***
# ***Fielding Factor=10, i.e. For each dismissal, the overall contribution of the player will be 10 ,***
# ***The sum of all these contributions will be used to calculate the overall score and to rank the best players of the Tournament.***
# 
# ***The main reason of keeping the Batting Factor so low as compared to the Bowling an f Fielding parameter is- the number of runs socred in a match is far more than the number of wickets that can be taken by the Teams. Hence inorder to make this list not too much inclined only towards batsmen, we have kept the factor to a very low Value. In the Top 10 we see 3 Batsmen,6 All Rounders and 1 Bowler making up into the list of the most utility players.***

# ## The Scores by an Entered list of Batsmen- through all the Seasons of IPL from 2008-2019.

# In[93]:


batsman_list_req = ['MS Dhoni','V Kohli','SC Ganguly']
batsman = df[df.batsman.isin(batsman_list_req)]
batsman_run = batsman.groupby(['match_id', 'batsman'])['batsman_runs'].sum().reset_index(name='Runs')

plt.figure(figsize=(30,10))
for name in batsman_list_req:
    batsman_check = batsman_run[batsman_run.batsman == name]
    batsman_check.index = np.arange(1, len(batsman_check) + 1)
    x = batsman_check.index
    y = batsman_check.Runs
    plt.bar(x, y, label=name)

plt.legend(prop={'size': 20})
plt.title("Innings Total across all Seasons- 2008-2019", fontsize=60)
plt.xlabel("Total Matches Played", fontsize=30)
plt.ylabel("Runs Scored in a Match", fontsize=30)
plt.show()


# ***Although the analysis can be varied if we enter the names of different players in the list- We have listed the 3 Captains of Indian Cricket- Sourav Ganguly, MS Dhoni and Virat Kohli, and monitored their batting performances. As clearly evident, MSD and Kohli have played much more than Ganguly, as Ganguly had taken a retirement much earlier. Also during the initial stages of IPL, Ganguly had a better performance matrix as compared to the rest- which eventually decreased to the point, where he retired from the game. Kohli and Dhoni have played comparable number of matches, with Kohli being a better performer. This also indicates his development and consistency in the Tournament.***

# ## Batsman innings wise performance against Opposition Team, The list of scores in individual matches and Economy rates of bowlers against the Batsman from the opposition Team.

# In[108]:


batsman_list_req = ['V Kohli']
opposition_team = 'Kolkata Knight Riders'
ball_limit = 12
cond_1_1 = df.batsman.isin(batsman_list_req)
cond_1_2 = df.bowling_team==opposition_team
batsman_team = df[(cond_1_1) & (cond_1_2)]

batsman_team_run = batsman_team.groupby(['match_id', 'batsman', 'bowling_team'])['batsman_runs'].sum().reset_index(name='Runs')

bowling_runs = batsman_team.groupby('bowler')['total_runs'].sum().reset_index(name='Runs Conceded')

bowling_balls = batsman_team.groupby('bowler')['ball'].count().reset_index(name='Balls Bowled')

bowled_balls_limit = bowling_balls[bowling_balls['Balls Bowled'] >= ball_limit]

bowler_stats = pd.merge(bowling_runs, bowled_balls_limit, on='bowler', how='inner')
bowler_stats['Economy Rate'] = (bowler_stats['Runs Conceded'] / bowler_stats['Balls Bowled']) * 6
bowler_best_to_worst = bowler_stats[['bowler', 'Balls Bowled', 'Economy Rate']].sort_values(by='Economy Rate')

plt.figure(figsize=(25,8))
plt.scatter(batsman_team_run.index, batsman_team_run.Runs, color='b')
plt.plot(batsman_team_run.index, batsman_team_run.Runs, 'r')
plt.title(f"{batsman_list_req[0]} innings wise score against {opposition_team} across 2008-2019", fontsize=30)
plt.xlabel("Total Matches Played", fontsize=30)
plt.ylabel("Runs Scored in a Match", fontsize=30)
plt.show()


print('The runs scored in matches:\n', batsman_team_run)
print('---------------------------------------------------------------------------------------------')
print(f'The Economy rate of the various bowlers of {opposition_team} against {batsman_list_req[0]} (best to worst)\n', bowler_best_to_worst)


# ***The Overall improvement of Virat Kohli as a Batsman can be seen in this chart. His performnace has clearly improved over time and age. Also from the given list of Bowlers, we can see Kohli has played exceptionally well against fast bowlers of the recent times, and had the lowest hittings for Slow pacers or Spinners- while in his early days.***
# 
# ***Additional Tip: As per the Current Team, Kolkata Knight Riders should try to restrict Kohli with their options of Spinners and Slow Pacers. Fast Bowlers would be a risk factor with Kohli on strike!***

# # Team Wise Analysis

# ## Innings wise batting average of the Teams

# In[124]:


first_innings_run = df[df['inning'] == 1]
team_innings_run = first_innings_run.groupby(['batting_team', 'match_id'])['total_runs'].sum().reset_index(name='Innings Total')
team_innings_avg = team_innings_run.groupby('batting_team')['Innings Total'].mean().reset_index(name='Innings Average')

plt.plot(team_innings_avg['batting_team'], team_innings_avg['Innings Average'], 'b', label='First Innings')

second_innings_run = df[df['inning'] == 2]
team_innings_run = second_innings_run.groupby(['batting_team', 'match_id'])['total_runs'].sum().reset_index(name='Innings Total')
team_innings_avg = team_innings_run.groupby('batting_team')['Innings Total'].mean().reset_index(name='Innings Average')

plt.plot(team_innings_avg['batting_team'], team_innings_avg['Innings Average'], 'r', label='Second Innings')

plt.xticks(rotation=90)
plt.legend(['First Innings', 'Second Innings'], prop={'size': 10})
plt.xlabel('IPL Teams',size=15)
plt.ylabel('Innings Average',size=15)
plt.title('Team wise Batting Average in IPL- Seasons 2008-2019',size=20)
plt.show()


# ## Innings wise bowling average of the Teams

# In[128]:


first_innings_score = df[df['inning'] == 1]

team_innings_score = first_innings_score.groupby(['bowling_team', 'match_id'])['total_runs'].sum().reset_index(name='Innings Total')

team_innings_score_avg = team_innings_score.groupby('bowling_team')['Innings Total'].mean().reset_index(name='Innings Average')

second_innings_score = df[df['inning'] == 2]

team_innings_second_score = second_innings_score.groupby(['bowling_team', 'match_id'])['total_runs'].sum().reset_index(name='Innings Total')

team_second_innings_score_avg = team_innings_second_score.groupby('bowling_team')['Innings Total'].mean().reset_index(name='Innings Average')

plt.plot(team_innings_score_avg['bowling_team'], team_innings_score_avg['Innings Average'], 'b')
plt.plot(team_second_innings_score_avg['bowling_team'], team_second_innings_score_avg['Innings Average'], 'r')

plt.xticks(rotation=90)
plt.legend(['First Innings', 'Second Innings'], prop={'size': 10})
plt.xlabel('IPL Teams', size=15)
plt.ylabel('Innings Average', size=15)
plt.title('Team wise Bowling Average in IPL - Seasons 2008-2019', size=20)
plt.show()


# ***One distinct observation that we have seen in this case is the second innings Batting and Bowling averages are less as compared to the firts innings.***

# ## Win by Runs/Win by Wickets - Team wise Average

# In[136]:


win_runs = df1.groupby('winner')['win_by_runs'].mean().reset_index(name='Win By Runs Average')
win_wickets = df1.groupby('winner')['win_by_wickets'].mean().reset_index(name='Win By Wickets Average')

plt.figure(figsize=(7,7))
plt.plot(win_runs['winner'], win_runs['Win By Runs Average'], color='b')
plt.plot(win_wickets['winner'], win_wickets['Win By Wickets Average'], color='r')

plt.xlabel('Teams', size=15)
plt.xticks(rotation=90)
plt.ylabel('Winning Metrics', size=15)
plt.legend(['Win by Runs', 'Win by Wickets'])
plt.title('Teams Average winning by Runs/Wickets Summary')


# ***While most teams have won matches with an average of 10+ runs- only 2 teams have shown values which are less than 5- Gujrat Lions and Kochi Tuskers Kerela. These teams have participated in limited seasons, and their results have not been succesful. But on a surprising note- they have won matches with an average wicket of 5+- which is a good sign in terms of the bowling attack of the team.***

# # Head to Head Match Analysis between the Teams of IPL

# In[148]:


Current_teams = ['Chennai Super Kings','Mumbai Indians','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad','Kolkata Knight Riders','Royal Challengers Bangalore','Kings XI Punjab']
team_1_filter = df1[df1.team1.isin(Current_teams)]
team_2_filter = team_1_filter[team_1_filter.team2.isin(Current_teams)]
teams_filter = team_2_filter[team_2_filter.winner.isin(Current_teams)]

head_to_head_matches = teams_filter.groupby(['team1', 'team2', 'winner']).size().reset_index(name='Winning Counts')
head_to_head_matches['Game'] = head_to_head_matches['team1'] + ' vs. ' + head_to_head_matches['team2']
head_to_head_matches.loc[:, ['Game', 'winner', 'Winning Counts']]

heatmap_data = pd.pivot_table(head_to_head_matches, values='Winning Counts', 
                             index=['Game'], 
                             columns='winner')
fig, ax = plt.subplots(figsize=(5, 18))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
ax.set_title('Head-to-Head Performance Matrix of Teams in IPL', size=15)
ax.set_xlabel('IPL Teams', size=15)
ax.set_ylabel('Match', size=15)
plt.show()


# # Team wise winning Performance at Venues

# In[158]:


venue_win = df1.groupby(['venue', 'winner']).size().reset_index(name='Match Wins')
venue_win_pvt = pd.pivot_table(venue_win, values='Match Wins', index='venue', columns='winner', fill_value=0)

plt.figure(figsize=(5, 15))
htmp = sns.heatmap(venue_win_pvt, annot=True, fmt='g', cmap='PuBuGn')
plt.xlabel('Teams', size=25)
plt.ylabel('Venues', size=25)
plt.title('Team wise wins at the Venues', size=30)
plt.show()


# ***The blue and green highlights are expected for the particular teams at the venues- For e.g. Chennai Super Kings at M.A. Chidambaram, Kolkata Knight Riders at Eden Gardens and Mumbai Indians at Wankhede Stadium-since the Venues being located at their home stadium, they have played more number of matches in that place. And since, more the number of games being played, more will obviously be the chances of winnings at the particular venue. But ignoring those, ie, when considering a teams capability to perform in the away matches, we see Mumbai Indians and Chennai Super Kings have shown very good performances in different venues, and that too multiple times. Undoubtedly, they have been the most succesful teams in the history of the tournament.***

# # Venue wise Best Performers

# In[161]:


venue_mom = df1.groupby(['venue', 'player_of_match']).size().reset_index(name='MoM_Winner')

venue_mom_sort = venue_mom.sort_values(by=['venue', 'MoM_Winner'], ascending=[True, False])

venue_mom_count_max = venue_mom_sort.groupby('venue', as_index=False).agg({'MoM_Winner': 'max'})

venue_best = pd.merge(venue_mom, venue_mom_count_max, on=['venue', 'MoM_Winner'], how='inner')

venue_best_multiple_pivot = venue_best.pivot(index='player_of_match', columns='venue', values='MoM_Winner')

plt.figure(figsize=(10, 15))
sns.heatmap(venue_best_multiple_pivot, annot=True, fmt='g', cmap='Wistia')
plt.xlabel('IPL Venues', size=25)
plt.ylabel('Players', size=25)
plt.title('Players with the Best Performance at Venues', size=20)
plt.show()


# ***Similar to the Team performances, Players have won more number of "Man of the Match" Trophies at their Home Ground. But some Players like AB de Villiers, Chris Gayle and MS Dhoni have performed significantly well in away Grounds as well.***
# 
# ***This metrics shows the reliability of a player when they are playing the game- irrespective of Home/Away Psychology. They can be relied upon as the key players in the team to perform.***

# # The Toss Decisions taken by Venue Heatmap-in IPL

# In[162]:


venue_toss = teams_filter.groupby(['venue', 'toss_decision']).size().reset_index(name='Toss Decision Counts')

heatmap2_data = pd.pivot_table(venue_toss, values='Toss Decision Counts', 
                               index=['venue'], 
                               columns='toss_decision')

fig, ax = plt.subplots(1, 1, figsize=(5, 15))
g = sns.heatmap(heatmap2_data, annot=True, cmap="Blues", fmt='g')
g.xaxis.set_ticks_position("top")
ax.set_title('The Toss Decisions taken by Venue Heatmap in IPL', size=20)
plt.show()


# # 

# # Most Winning Teams

# In[173]:


winner_team = df1['winner'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=winner_team, y=winner_team.index)
plt.xticks(rotation = 75)
plt.title('winning team')
plt.xlabel('Count')


# # Match Results

# In[174]:


rs = df1['result'].value_counts()


# In[175]:


rs.plot(kind='bar')


# # Most Matches Played At Venues

# In[176]:


df1['venue'].value_counts().reset_index()


# In[ ]:




