from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
xgb.set_config(verbosity=0)










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