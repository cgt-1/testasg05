import pandas as pd

def feature_engineering(df):
    df = df.copy()

    
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Cabin_num'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]

    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')
    df['Solo'] = (df['Group_size'] == 1).astype(int)

    df['TotalSpending'] = df[
        ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    ].sum(axis=1)

    return df