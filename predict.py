import argparse

import joblib
import pandas as pd

from utils import get_df_per_location, create_lagged_feature, fill_top_rows_from_historic_last_rows, get_lagged_col_name


def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    models = joblib.load(model_fn)
    future_per_location = get_df_per_location(future_climatedata_fn)
    historic_per_location = get_df_per_location(historic_data_fn)
    first_location = True
    for location in future_per_location.keys():
        df = future_per_location[location]
        historic_df = historic_per_location[location]
        model = models[location]

        X = df[['rainfall', 'mean_temperature']]
        create_lagged_feature(X, 'mean_temperature', 1)
        create_lagged_feature(X, 'rainfall', 1)
        fill_top_rows_from_historic_last_rows('mean_temperature', 1, X, historic_df)
        fill_top_rows_from_historic_last_rows('rainfall', 1, X, historic_df)

        #note: while the predictions will be put in a column sample_0,
        #the lagged columns need to be named diasese_cases to match features in trained model
        df['sample_0'] = pd.NA
        last_disease_col = get_lagged_col_name('disease_cases', 1)
        X[last_disease_col] = pd.NA
        # features = ['rainfall', 'mean_temperature', 'last_rainfall', 'last_disease_cases']

        prev_disease = historic_df['disease_cases'].iloc[-1]
        for i in range(X.shape[0]):
            X.loc[i,last_disease_col] = prev_disease
            y_one_pred = model.predict(X.iloc[i:i+1])
            df.loc[i,'sample_0'] = y_one_pred

            prev_disease = y_one_pred

        if first_location:
            df.to_csv(predictions_fn, index=False, mode='w', header=True)
            first_location = False
        else:
            df.to_csv(predictions_fn, index=False, mode='a', header=False)
        print("predict - forecast values: ", df['sample_0'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
