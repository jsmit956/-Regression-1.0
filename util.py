from sklearn.model_selection import TimeSeriesSplit


def train_test_time(df,target):
    """
    df:Dataframe of features
    target:str of column to regression on
    """
    y=df[target]
    tickers=["SPY","QQQ","VTV","VBR"]
    X = df.drop(tickers,axis="columns") 
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    for train_index, test_index in tscv.split(X):
        #print(train_index)
        #print(test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train,X_test,y_train,y_test

