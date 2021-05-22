import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
xgb.set_config(verbosity=0)




def missing_values(df):
    imp = SimpleImputer(strategy="most_frequent")
    imp_fit = imp.fit_transform(df)
    tmp = df.columns
    df = pd.DataFrame(imp_fit,columns=tmp)
    df['retail_and_recreation_percent_change_from_baseline'] = pd.to_numeric(df['retail_and_recreation_percent_change_from_baseline'])
    df['grocery_and_pharmacy_percent_change_from_baseline'] = pd.to_numeric(df['grocery_and_pharmacy_percent_change_from_baseline'])
    df['parks_percent_change_from_baseline'] = pd.to_numeric(df['parks_percent_change_from_baseline'])
    df['transit_stations_percent_change_from_baseline'] = pd.to_numeric(df['transit_stations_percent_change_from_baseline'] )
    df['workplaces_percent_change_from_baseline'] = pd.to_numeric(df['workplaces_percent_change_from_baseline'])
    df['residential_percent_change_from_baseline'] = pd.to_numeric(df['residential_percent_change_from_baseline'])
    df['year'] = pd.to_numeric(df['year'])
    df['month'] = pd.to_numeric(df['month'])
    df['day'] = pd.to_numeric(df['day'])
    df["mobility"] = pd.to_numeric(df["mobility"])



    print("\nMissing values appended, strategy: most_frequent")

    return df



def data_preprocess():
    subregion_scores = {}
    df = pd.read_csv("2020_TR_Region_Mobility_Report.csv")
    # print(df.info())

    df = df[df["sub_region_1"] == "İstanbul"]
    df = df.drop(["metro_area", "iso_3166_2_code", "census_fips_code", "place_id"], axis=1)
    df = df[df["sub_region_2"] != "Adalar"]
    df = df.dropna(subset=["sub_region_2"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day

    seaborn.lineplot(data = df,x = 'month',y ='retail_and_recreation_percent_change_from_baseline' ,color="red")
    plt.show()
    seaborn.lineplot(data=df, x='month', y='grocery_and_pharmacy_percent_change_from_baseline', color="red")
    plt.show()
    seaborn.lineplot(data=df, x='month', y='parks_percent_change_from_baseline', color="red")
    plt.show()
    seaborn.lineplot(data=df, x='month', y='transit_stations_percent_change_from_baseline', color="red")
    plt.show()
    seaborn.lineplot(data=df, x='month', y='workplaces_percent_change_from_baseline', color="red")
    plt.show()
    seaborn.lineplot(data=df, x='month', y='residential_percent_change_from_baseline', color="red")
    plt.show()
    seaborn.histplot(df['retail_and_recreation_percent_change_from_baseline'], color="red")
    plt.show()
    seaborn.histplot(df['grocery_and_pharmacy_percent_change_from_baseline'], color="red")
    plt.show()
    seaborn.histplot(df['parks_percent_change_from_baseline'], color="red")
    plt.show()
    seaborn.histplot(df['transit_stations_percent_change_from_baseline'], color="red")
    plt.show()
    seaborn.histplot(df['workplaces_percent_change_from_baseline'], color="red")
    plt.show()
    seaborn.histplot(df['residential_percent_change_from_baseline'], color="red")
    plt.show()

    df["mobility"] = df['retail_and_recreation_percent_change_from_baseline'] + df['grocery_and_pharmacy_percent_change_from_baseline'] + df['parks_percent_change_from_baseline'] + df['transit_stations_percent_change_from_baseline'] + df['workplaces_percent_change_from_baseline'] + df['residential_percent_change_from_baseline']

    seaborn.lineplot(data=df, x='month', y="mobility", color="red")
    plt.show()
    seaborn.histplot(df["mobility"], color="red")
    plt.show()

    for model in df['sub_region_2'].value_counts().iteritems():
        #print(model[0], ": ", model[1])
        subregion_scores.setdefault(model[0],0)


    df = missing_values(df)

    print(df.info())
    print(df.head())
    print(df.describe())


    return df,subregion_scores




def prepare_for_train(df,sub_region):
    df = df[df["sub_region_2"] == sub_region]
    X = df.drop(["country_region_code","country_region","sub_region_1","sub_region_2","date","mobility"],axis = 1)
    y = df["mobility"]
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=20,train_size=80)

    return x_train,x_test,y_train,y_test


def train_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBRegressor(n_jobs=1,booster="gblinear",max_depth=2).fit(x_train,y_train)
    pred = xgb_model.predict(x_test)
    score = r2_score(y_test,pred)*100


    return score


def xgbooster(df,subregion_scores):
    for sub_region in subregion_scores:
        x_train, x_test, y_train, y_test = prepare_for_train(df, sub_region)
        score = train_xgboost(x_train, x_test, y_train, y_test)
        subregion_scores[sub_region] = score

    return subregion_scores

if __name__ == '__main__':
    df,subregion_scores = data_preprocess()
    subregion_scores = xgbooster(df,subregion_scores)
    for data in subregion_scores:
        print(data, " prediction boost score is: %" , subregion_scores[data])
