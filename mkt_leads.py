# read the data and set the datetime as the index
import pandas as pd
#df = pd.read_csv(r"C:\Users\nfrq38\DS\Prj\data_set_inquiries_CN.csv",dayfirst=True, parse_dates=[0])
df = pd.read_csv("https://raw.githubusercontent.com/gusharani/project_DS/master/data_set_inquiries_CN.csv",dayfirst=True, parse_dates=[0])
df.head()




df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['Year'] = df.index.year
df['Month'] = df.index.month




# Filter to month where there is email promo
import seaborn as sb
email_promo_y = df[df.Email_Promo==1]
email_promo_y[['Est_Rev']].plot()

email_promo_n = df[df.Email_Promo==0]
email_promo_n[['Leads']].plot()

# Filter to month where there is road show
road_show_y = df[df.Road_Show==1]
road_show_y[['Leads']].plot()

road_show_n = df[df.Road_Show==0]
road_show_n[['Leads']].plot()



# Filter to month where there is road show
event_y = df[df.Event==1]
event_y[['Leads']].plot()

event_n = df[df.Event==0]
event_n[['Leads']].plot()


# Filter to month where there is road show
ads_y = df[df.Display_Ads==1]
ads_y[['Leads']].plot()

ads_n = df[df.Display_Ads==0]
ads_n[['Leads']].plot()

# Quartely sales

###########

# Filter to store 1 sales and average over weeks
email_1_rev = df[df.Email_Promo == 1][['Est_Rev']].resample('M', 'sum')
email_1_rev.head()

pd.rolling_mean(email_1_rev[['Est_Rev']], 3).plot()


print('Autocorrelation 1: ', email_1_rev['Est_Rev'].autocorr(1))
print('Autocorrelation 3: ', email_1_rev['Est_Rev'].autocorr(3))
print('Autocorrelation 52: ', email_1_rev['Est_Rev'].autocorr(52))


from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(email_1_rev['Est_Rev'])


from statsmodels.graphics.tsaplots import plot_acf

plot_acf(email_1_rev['Est_Rev'], lags=10)

n = len(email_1_rev.Est_Rev)

train = email_1_rev.Est_Rev[:int(.75*n)]
test = email_1_rev.Est_Rev[int(.75*n):]



import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

model = sm.tsa.ARIMA(train, (1, 0, 0)).fit()

predictions = model.predict(
    pd.to_datetime('01/31/2017'),
    pd.to_datetime('28/02/2017'),
    dynamic=True,
)

print("Mean absolute error: ", mean_absolute_error(test, predictions))
model.summary()



