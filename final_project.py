import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

############################ QUESTION 1 ##########################################

def trading_strategy(df,window):
    df['alpha']=0
    df['beta']=0
    df['t_value']=0
    df['alpha+beta*Xt']=0
    df['AchatEuro']=0
    df['AchatDollar']=0
    df['NbreEuro']=0
    df['Portefeuille']=1
    df['Portefeuille'].iloc[:window]=1
    df['Returns']=df['exchange_rate'].pct_change()#PourTest
    df['Cumulative_Return']=1#PourTest
    df['Strategy_Return']=0#PourTest
    
    for n in range(window,len(df)-1):
        dfwindow=df.iloc[n-window:n].copy()
        model=sm.OLS(dfwindow['X(T+DeltaT)-X(T)'],sm.add_constant(dfwindow['exchange_rate'])).fit()
        df['alpha'].iloc[n]=model.params[0]
        df['beta'].iloc[n]=model.params[1]
        df['t_value'].iloc[n]=model.tvalues[1]
        df['alpha+beta*Xt'].iloc[n]=df['alpha'].iloc[n]+df['beta'].iloc[n]*df['exchange_rate'].iloc[n]
        
        if df['t_value'].iloc[n]<-1 and df['alpha'].iloc[n]>0 and df['beta'].iloc[n]<0:
            if df['alpha+beta*Xt'].iloc[n]>0:
                df['AchatEuro'].iloc[n+1]=1##Fait des sous avec n+1
                df['AchatDollar'].iloc[n+1]=0##Fait des sous avec n+1
            else:
                df['AchatEuro'].iloc[n+1]=0##Fait des sous avec n+1
                df['AchatDollar'].iloc[n+1]=1##Fait des sous avec n+1
        else:
            df['AchatEuro'].iloc[n+1] = 0
            df['AchatDollar'].iloc[n+1] = 0#test pour dire qu'on passe en dollar si aucune condition remplie, change rien
            
                    
        if df['AchatEuro'].iloc[n+1]==1 and df['AchatEuro'].iloc[n]==0:
            df['NbreEuro'].iloc[n+1]=df['Portefeuille'].iloc[n]/df['exchange_rate'].iloc[n]##n-1 plutot que n, test
        else:
            df['NbreEuro'].iloc[n+1]=df['NbreEuro'].iloc[n]
    
        if df['AchatEuro'].iloc[n+1]==1:
            df['Portefeuille'].iloc[n+1]=df['NbreEuro'].iloc[n+1]*df['exchange_rate'].iloc[n+1]
        else:
            df['Portefeuille'].iloc[n+1]=df['Portefeuille'].iloc[n]
        df['Strategy_Return'].iloc[n]=df['Returns'].iloc[n]*df['AchatEuro'].iloc[n]-df['Returns'].iloc[n]*df['AchatDollar'].iloc[n]#PourTest
        df['Cumulative_Return'].iloc[n]=df['Cumulative_Return'].iloc[n-1]*(1+df['Strategy_Return'].iloc[n])
        
    df['PortefeuilleEURO']=df['Portefeuille']*df['exchange_rate']
    df['Rendement'] = (1 + df['Portefeuille'].pct_change()).cumprod()
    df['Rendement'] = df['Rendement']-1
        
    return df.iloc[:-1]

#f1x=tradingStrat(dfx,5)
##RecupererDonnées        
eurusd=yf.Ticker('EURUSD=X')
start_date=datetime.date(2010,1,1)
end_date=datetime.date(2024,12,31)
df=eurusd.history(start=start_date,end=end_date)
df.drop(["Volume", "Dividends","Stock Splits","Open","High"], axis=1, inplace=True)
df['exchange_rate']=df['Close']
df['X(T+DeltaT)-X(T)']=df['exchange_rate'].shift(-1)-df['exchange_rate'].dropna()

##TestFullSample
model=sm.OLS(df['X(T+DeltaT)-X(T)'],sm.add_constant(df['exchange_rate'])).fit()
alpha=model.params[0]
beta=model.params[1]
t_value=model.tvalues[1]
##TestSample

##Question2
window=80

df2=trading_strategy(df,window)
df2.index = pd.to_datetime(df2.index)
plt.plot(df2.index,df2['Portefeuille'],'b',label='date')
plt.legend()
plt.title("window"+str(window))
plt.show()
plt.plot(df2.index,df2['Cumulative_Return'],'b',label='date')#PourTest
plt.legend()
plt.title("window"+str(window)+str("methode sans monnaie"))
plt.show()
###RatioSharpe
####Appliqué au graphique1, donc à 1 dollar
risk_free_rate_annual = 0.02  # 2% annuel
risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1/252) - 1
mean_returnD = df2['Portefeuille'].pct_change().mean()
volatilityD = df2['Portefeuille'].pct_change().std()
sharpe_ratioD = (mean_returnD-risk_free_rate_daily)/volatilityD
sharpe_ratio_annualizedD = sharpe_ratioD * np.sqrt(252)
####Appliqué au graphique2, donc sans monnaie
mean_return = df2['Returns'].mean()
volatility = df2['Returns'].std()
sharpe_ratio = (mean_return-risk_free_rate_daily)/volatility
sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)


############################ QUESTION 3 ############################################

def trading_strategy2(df,windows):
    df['NbreEuro']=0
    df['Portefeuille']=1
    df['Returns']=df['euro/dollar'].pct_change()#PourTest
    df['Cumulative_Return']=1
    df['Strategy_Return']=0
    
    for i in windows:
        df['alpha'+str(i)]=0
        df['beta'+str(i)]=0
        df['t_value'+str(i)]=0
        df['alpha+beta*Xt'+str(i)]=0
        df['AchatEuro'+str(i)]=0
        df['AchatDollar'+str(i)]=0
        df['sum_achat_euro']=0
        df['sum_achat_dollar']=0
        window = i
        for n in range(window,len(df)-1):
            dfwindow=df.iloc[n-window:n].copy()
            model=sm.OLS(dfwindow['X(T+DeltaT)-X(T)'],sm.add_constant(dfwindow['euro/dollar'])).fit()
            df['alpha'+str(i)].iloc[n]=model.params[0]
            df['beta'+str(i)].iloc[n]=model.params[1]
            df['t_value'+str(i)].iloc[n]=model.tvalues[1]
            df['alpha+beta*Xt'+str(i)].iloc[n]=df['alpha'+str(i)].iloc[n]+df['beta'+str(i)].iloc[n]*df['euro/dollar'].iloc[n]
            
            if df['t_value'+str(i)].iloc[n]<-1 and df['alpha'+str(i)].iloc[n]>0 and df['beta'+str(i)].iloc[n]<0:
                if df['alpha+beta*Xt'+str(i)].iloc[n]>0:
                    df['AchatEuro'+str(i)].iloc[n+1]=1
                    df['AchatDollar'+str(i)].iloc[n+1]=0
                else:
                    df['AchatEuro'+str(i)].iloc[n+1]=0
                    df['AchatDollar'+str(i)].iloc[n+1]=1
            else:
                df['AchatEuro'+str(i)].iloc[n+1] = 0
                df['AchatDollar'+str(i)].iloc[n+1] = 0
                        
    for i in windows:
        window = i
        for n in range(window,len(df)-1):
            df['sum_achat_euro'].iloc[n] += df['AchatEuro'+str(i)].iloc[n]
            df['sum_achat_dollar'].iloc[n] += df['AchatDollar'+str(i)].iloc[n]
    
    for n in range(20, len(df)-1):
        if df['sum_achat_euro'].iloc[n] == min_signal and df['sum_achat_dollar'].iloc[n] == min_signal:
            bool_achat_eur = False
            bool_achat_dollar = False
        elif df['sum_achat_euro'].iloc[n]>= min_signal and df['sum_achat_dollar'].iloc[n]<min_signal:
            df['NbreEuro'].iloc[n+1]=df['Portefeuille'].iloc[n]/df['euro/dollar'].iloc[n]
            bool_achat_eur = True
            bool_achat_dollar = False
        elif df['sum_achat_euro'].iloc[n]<min_signal and df['sum_achat_dollar'].iloc[n]>=min_signal:
            bool_achat_eur = False
            bool_achat_dollar = True
        else:
            df['NbreEuro'].iloc[n+1]=df['NbreEuro'].iloc[n]
            bool_achat_eur = False
            bool_achat_dollar = False
            
        if bool_achat_eur:
            df['Portefeuille'].iloc[n+1]=df['NbreEuro'].iloc[n+1]*df['euro/dollar'].iloc[n+1]
        else:
            df['Portefeuille'].iloc[n+1]=df['Portefeuille'].iloc[n]
        df['Strategy_Return'].iloc[n]=df['Returns'].iloc[n]*bool_achat_eur-df['Returns'].iloc[n]*bool_achat_dollar
        df['Cumulative_Return'].iloc[n]=df['Cumulative_Return'].iloc[n-1]*(1+df['Strategy_Return'].iloc[n])
    
    df['PortefeuilleEURO']=df['Portefeuille']*df['euro/dollar']
    df['Rendement'] = (1 + df['Portefeuille'].pct_change()).cumprod()
    df['Rendement'] = df['Rendement']-1
        
    return df.iloc[:-1]

eurusd=yf.Ticker('EURUSD=X')
start_date=datetime.date(2010,1,1)
end_date=datetime.date(2024,12,31)
df=eurusd.history(start=start_date,end=end_date)
df.drop(["Volume", "Dividends","Stock Splits","Open","High"], axis=1, inplace=True)
df['euro/dollar']=df['Close']
df['X(T+DeltaT)-X(T)']=df['euro/dollar'].shift(-1)-df['euro/dollar']
df=df[['euro/dollar','X(T+DeltaT)-X(T)']]

windows=[30,100]
min_signal = len(windows)/2

df2=trading_strategy2(df,windows)
df2.index = pd.to_datetime(df2.index)
#plt.plot(df2.index,df2['Portefeuille'],'b',label='date')
#plt.legend()
#plt.title("window")
#plt.show()
plt.plot(df2.index,df2['Cumulative_Return'],'b',label='date')#PourTest
plt.legend()
plt.title("Cumulative Absolute Returns")
plt.show()

############################ QUESTION 4 A ##########################################


pairs = ["EURUSD=X", "EURGBP=X", "GBPUSD=X", "JPYUSD=X"]

start_date = datetime.date(2010, 1, 1)
end_date = datetime.date(2024, 12, 31)

window = 80

def analyze_pair(pair):
    ticker = yf.Ticker(pair)
    df = ticker.history(start=start_date, end=end_date)

    df.drop(["Volume", "Dividends", "Stock Splits", "Open", "High"], axis=1, inplace=True)
    df['exchange_rate'] = df['Close']
    df['X(T+DeltaT)-X(T)'] = df['exchange_rate'].shift(-1) - df['exchange_rate']
    df = df[['exchange_rate', 'X(T+DeltaT)-X(T)']]

    # Application de la stratégie
    df2 = trading_strategy(df, window)
    df2.index = pd.to_datetime(df2.index)

    plt.figure(figsize=(10, 5))
    plt.plot(df2.index, df2['Portefeuille'], 'b', label='Portefeuille')
    plt.legend()
    plt.title(f"{pair} - 1 dollar Evolution")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df2.index, df2['Cumulative_Return'], 'b', label='Cumulative Return')
    plt.legend()
    plt.title(f"{pair} - Cumulative Rturns")
    plt.show()

for pair in pairs:
    print(f"Analyse de {pair}")
    analyze_pair(pair)

############################ QUESTION 4 B ##########################################

warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_parameters(df,window):
    df['alpha']=0
    df['beta']=0
    df['t_value']=0
    df['alpha+beta*Xt']=0
    df['Returns']=df['exchange_rate'].pct_change()
    
    for n in range(window,len(df)-1):
        dfwindow=df.iloc[n-window:n].copy()
        model=sm.OLS(dfwindow['X(T+DeltaT)-X(T)'],sm.add_constant(dfwindow['exchange_rate'])).fit()
        df['alpha'].iloc[n]=model.params[0]
        df['beta'].iloc[n]=model.params[1]
        df['t_value'].iloc[n]=model.tvalues[1]
        df['alpha+beta*Xt'].iloc[n]=df['alpha'].iloc[n]+df['beta'].iloc[n]*df['exchange_rate'].iloc[n]
        
    return df.iloc[:-1]

def get_max_expected_change(dataframes,window):
    df_final['AchatDevise']=0
    df_final['AchatDollar']=0
    df_final['Expected_change1']=0
    df_final['Expected_change2']=0
    df_final['Expected_change3']=0
    df_final['Max_expected_change']=0
    df_final['Pair']=0
    df_final['NbreDevise']=0
    df_final['Portefeuille']=1
    df_final['Portefeuille'].iloc[:window]=1
    df_final['Cumulative_Return']=1
    df_final['Strategy_Return']=0
    max_expected_change = 0
    
    for n in range(window,len(dataframes[0])-1):
        df_final['Expected_change1'].iloc[n]=dataframes[0]['alpha+beta*Xt'].iloc[n].item()
        df_final['Expected_change2'].iloc[n]=dataframes[1]['alpha+beta*Xt'].iloc[n].item()
        df_final['Expected_change3'].iloc[n]=dataframes[2]['alpha+beta*Xt'].iloc[n].item()
        for i in range(len(dataframes)):
            #print("ICI")
            #print(abs(dataframes[i]['alpha+beta*Xt'].iloc[n].item()))
            if max_expected_change < abs(dataframes[i]['alpha+beta*Xt'].iloc[n].item()):
                max_expected_change = abs(dataframes[i]['alpha+beta*Xt'].iloc[n].item())
                ind_best_pair = i
                df_final['Pair'].iloc[n+1] = i
                df_final['Max_expected_change'].iloc[n+1] = max_expected_change
        if dataframes[ind_best_pair]['t_value'].iloc[n]<-1 and dataframes[ind_best_pair]['alpha'].iloc[n]>0 and dataframes[ind_best_pair]['beta'].iloc[n]<0:
            if dataframes[ind_best_pair]['alpha+beta*Xt'].iloc[n]>0:
                df_final['AchatDevise'].iloc[n+1]=1
                df_final['AchatDollar'].iloc[n+1]=0
            else:
                df_final['AchatDevise'].iloc[n+1]=0
                df_final['AchatDollar'].iloc[n+1]=1
        else:
            df_final['AchatDevise'].iloc[n+1] = 0
            df_final['AchatDollar'].iloc[n+1] = 0
            
                    
        if df_final['AchatDevise'].iloc[n+1]==1 and df_final['AchatDevise'].iloc[n]==0:
            df_final['NbreDevise'].iloc[n+1]=df_final['Portefeuille'].iloc[n]/dataframes[ind_best_pair]['exchange_rate'].iloc[n]
        else:
            df_final['NbreDevise'].iloc[n+1]=df_final['NbreDevise'].iloc[n]
    
        if df_final['AchatDevise'].iloc[n+1]==1:
            df_final['Portefeuille'].iloc[n+1]=df_final['NbreDevise'].iloc[n+1]*dataframes[ind_best_pair]['exchange_rate'].iloc[n+1]
        else:
            df_final['Portefeuille'].iloc[n+1]=df_final['Portefeuille'].iloc[n]
        df_final['Strategy_Return'].iloc[n]=dataframes[ind_best_pair]['Returns'].iloc[n]*df_final['AchatDevise'].iloc[n]-dataframes[ind_best_pair]['Returns'].iloc[n]*df_final['AchatDollar'].iloc[n]#PourTest
        df_final['Cumulative_Return'].iloc[n]=df_final['Cumulative_Return'].iloc[n-1]*(1+df_final['Strategy_Return'].iloc[n])
        
    df_final['Rendement'] = (1 + df_final['Portefeuille'].pct_change()).cumprod()
    df_final['Rendement'] = df_final['Rendement']-1
                
    return df_final.iloc[:-1]

eurusd=yf.Ticker('EURUSD=X')
gbpusd=yf.Ticker('GBPUSD=X')
jpyusd=yf.Ticker('JPYUSD=X')
eurgbp = yf.Ticker('EURGBP=X')

start_date=datetime.date(2010,1,1)
end_date=datetime.date(2024,12,31)

df_eurusd = eurusd.history(start=start_date,end=end_date)
df_gbpusd = gbpusd.history(start=start_date,end=end_date)
df_jpyusd = jpyusd.history(start=start_date,end=end_date)
df_eurgbp = jpyusd.history(start=start_date,end=end_date)


df_eurusd.drop(["Volume", "Dividends","Stock Splits","Open","High"], axis=1, inplace=True)
df_eurusd['exchange_rate']=df_eurusd['Close']
df_eurusd['X(T+DeltaT)-X(T)']=df_eurusd['exchange_rate'].shift(-1)-df_eurusd['exchange_rate']
df_eurusd=df_eurusd[['exchange_rate','X(T+DeltaT)-X(T)']]

df_gbpusd.drop(["Volume", "Dividends","Stock Splits","Open","High"], axis=1, inplace=True)
df_gbpusd['exchange_rate']=df_gbpusd['Close']
df_gbpusd['X(T+DeltaT)-X(T)']=df_gbpusd['exchange_rate'].shift(-1)-df_gbpusd['exchange_rate']
df_gbpusd=df_gbpusd[['exchange_rate','X(T+DeltaT)-X(T)']]

df_eurgbp.drop(["Volume", "Dividends","Stock Splits","Open","High"], axis=1, inplace=True)
df_eurgbp['exchange_rate']=df_eurgbp['Close']
df_eurgbp['X(T+DeltaT)-X(T)']=df_eurgbp['exchange_rate'].shift(-1)-df_eurgbp['exchange_rate']
df_eurgbp=df_eurgbp[['exchange_rate','X(T+DeltaT)-X(T)']]

df_jpyusd.drop(["Volume", "Dividends","Stock Splits","Open","High"], axis=1, inplace=True)
df_jpyusd['exchange_rate']=df_jpyusd['Close']
df_jpyusd['X(T+DeltaT)-X(T)']=df_jpyusd['exchange_rate'].shift(-1)-df_jpyusd['exchange_rate']
df_jpyusd=df_jpyusd[['exchange_rate','X(T+DeltaT)-X(T)']]

df_final = pd.DataFrame(df_eurusd.index)
df_final.columns = ['Date']

window=80

df_eurusd = compute_parameters(df_eurusd,window)
df_gbpusd = compute_parameters(df_gbpusd,window)
df_jpyusd = compute_parameters(df_jpyusd,window)
df_eurgbp = compute_parameters(df_eurgbp,window)


list_dataframes = [df_eurusd, df_gbpusd, df_jpyusd]
df_final = get_max_expected_change(list_dataframes, window)
df_final = df_final.drop(df_final.index[-1])


#plt.plot(df_final['Date'],df_final['Portefeuille'],'b',label='date')
#plt.legend()
#plt.title("Graphe 1")
#plt.show()
plt.plot(df_final['Date'],df_final['Cumulative_Return'],'b',label='date')
plt.legend()
plt.title("Cumulative Absolute Returns")
plt.show()