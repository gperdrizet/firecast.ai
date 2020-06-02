import pandas as pd
from multiprocessing import Pool

import CA_only.config as config

from CA_only.weather_data_helper_functions import collect_by_year


def combine_weather_data(data_years: list):
    with Pool(config.COLLECT_PROCESSES) as pool:
        pool.map(collect_by_year, data_years)

    pool.close()
    pool.join()

    df_list = []
    # append each dataframe to list
    for year in data_years:
        input_file = (config.WEATHER_DATA_BASE_PATH + config.COMPLETE_PARSED_DATAFILE_SUBDIR + "all." +
                      str(year) + ".california_only.parquet")

        df = pd.read_parquet(input_file)
        df_list.append(df)

    # concatenate list of dataframes
    result = pd.concat(df_list)

    # write to disk
    # output_file = (config.WEATHER_DATA_BASE_PATH + "all." +
    #                str(config.START_YEAR) + "-" + str(config.END_YEAR) + ".california_only.parquet")

    # result.to_parquet(output_file)

    return result
