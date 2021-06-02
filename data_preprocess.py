import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.impute import SimpleImputer







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
    print(df.info())

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

