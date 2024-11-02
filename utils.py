import pandas as pd

#Note: these functions are here copied in from the repository chap_model_dev_toolkit to make this example repository easier to develop self-contained.
#For real development, we recommend installing chap_model_dev_toolkit and instead importing these function from there.
def get_df_per_location(csv_fn: str) -> dict:
    full_df = pd.read_csv(csv_fn)
    unique_locations_list = full_df['location'].unique()
    locations = {location: full_df[full_df['location'] == location].reset_index(drop=True) for location in unique_locations_list}
    return locations

def get_lagged_col_name(feature, lag) -> str:
    return f'{feature}_lag_{lag}'

def create_lagged_feature(df_to_change, feature, num_lags, source_df=None):
    if source_df is None:
        source_df = df_to_change
    lag_features = []
    for lag in range(1, num_lags + 1):
        lag_features.append( get_lagged_col_name(feature, lag) )
        df_to_change[ lag_features[-1]] = source_df[feature].shift(lag)
    return lag_features


def cut_top_rows(df, int_remove_rows):
    return df.iloc[int_remove_rows:].reset_index(drop=True)

def fill_top_rows_from_historic_last_rows(feature, lag, future_df, historic_df):
    for row_index in range(lag):
        future_df.loc[row_index, get_lagged_col_name(feature, lag) ] = historic_df[ feature ].iloc[-(lag-row_index)]

