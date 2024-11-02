import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import get_df_per_location, create_lagged_feature, cut_top_rows


#For strategies that train separate models per region, this function can be used as is
def train(csv_fn, model_fn):
    models = {}
    locations = get_df_per_location(csv_fn)
    for location,df in locations.items():
        model = train_single_region(df, location)
        models[location] = model

    joblib.dump(models, model_fn)


def train_single_region(df, location):
    #features = ['rainfall', 'mean_temperature', 'last_rainfall', 'last_disease_cases']
    features = ['rainfall', 'mean_temperature']
    X = df[features]
    Y = df['disease_cases']
    # X['last_rainfall'] = df['rainfall'].shift(1)
    # X['last_disease_cases'] = df['disease_cases'].shift(1)
    create_lagged_feature(X, 'mean_temperature', 1)
    create_lagged_feature(X, 'rainfall', 1)
    create_lagged_feature(X, 'disease_cases', 1, df)
    X = cut_top_rows(X, 1)
    Y = cut_top_rows(Y, 1)

    model = LinearRegression()
    model.fit(X, Y)
    # model.last_disease_cases = Y.iloc[-1]
    # model.last_rainfall = X['rainfall'].iloc[-1]
    print(f"Train - model coefficients for location {location}: ", list(zip(features, model.coef_)))
    return model







