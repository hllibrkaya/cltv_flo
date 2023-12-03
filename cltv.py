import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import warnings

warnings.filterwarnings("ignore")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Finds categoric, cardinal and numeric column names in given dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame.
    cat_th : int, optional
        Categorical threshold for unique values, default is 10.
    car_th : int, optional
        Cardinality threshold for unique values, default is 20.

    Returns
    -------
    tuple
        Tuple containing lists of categorical columns, categorical but cardinal columns, and numerical columns.

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, cat_but_car, num_cols


def outlier_thresholds(dataframe, column, low_quantile=0.01, up_quantile=0.99):
    """
    Calculates the lower and upper outlier thresholds for a given column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame.
    column : str
        The column name for which to calculate thresholds.
    low_quantile : float, optional
        The lower quantile for threshold calculation, default is 0.01.
    up_quantile : float, optional
        The upper quantile for threshold calculation, default is 0.99.

    Returns
    -------
    tuple
        Tuple containing the lower and upper thresholds.

    """
    quantile1 = dataframe[column].quantile(low_quantile)
    quantile3 = dataframe[column].quantile(up_quantile)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, column):
    """
     Replaces values outside the specified thresholds with rounded threshold values.

    Parameters
    ----------
    dataframe : pd.Dataframe
        The input DataFrame.
    column : str
        The column name for threshold replacement.

    Returns
    -------
    None

    """
    low_limit, up_limit = outlier_thresholds(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = round(low_limit, 0)
    dataframe.loc[(dataframe[column] > up_limit), column] = round(up_limit, 0)


def data_prep(dataframe):
    """
    Prepares the data by handling outliers and creating additional features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The prepared DataFrame.

    """
    _, _, numeric = grab_col_names(dataframe)
    for col in numeric:
        replace_with_thresholds(dataframe, col)

    dataframe["total_order_num"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    date_cols = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)
    return dataframe


def create_cltv(dataframe):
    """
    Creates Customer Lifetime Value (CLTV) features based on the input DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing CLTV features.

    """
    dataframe = data_prep(dataframe)
    last_date = dataframe["last_order_date"].max()
    analysis_date = last_date + pd.Timedelta(days=2)

    cltv = pd.DataFrame()
    cltv["customer_id"] = df["master_id"]
    cltv["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
    cltv["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7
    cltv["frequency"] = df["total_order_num"]
    cltv["monetary_cltv_avg"] = df["customer_value_total"] / cltv["frequency"]

    return cltv


def modelling(dataframe, month=6):
    """
    Performs CLTV modeling using BetaGeoFitter and GammaGammaFitter.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame.
    month : int, optional
        The number of months for CLTV prediction, default is 6.

    Returns
    -------
    pd.DataFrame
        DataFrame containing CLTV model predictions for the specified number of months.
    """
    dataframe = create_cltv(dataframe)
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(dataframe["frequency"], dataframe["recency_cltv_weekly"], dataframe["T_weekly"])

    dataframe["exp_sales_3_month"] = bgf.predict(12, dataframe["frequency"], dataframe["recency_cltv_weekly"],
                                                 dataframe["T_weekly"])
    dataframe[f"exp_sales_{month}_month"] = bgf.predict(month * 4, dataframe["frequency"],
                                                        dataframe["recency_cltv_weekly"],
                                                        dataframe["T_weekly"])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(dataframe["frequency"], dataframe["monetary_cltv_avg"])

    dataframe["exp_average_value"] = ggf.conditional_expected_average_profit(dataframe["frequency"])

    dataframe["cltv"] = ggf.customer_lifetime_value(bgf, dataframe["frequency"], dataframe["recency_cltv_weekly"],
                                                    dataframe["T_weekly"], dataframe["monetary_cltv_avg"],
                                                    time=month, freq="W", discount_rate=0.01)

    return dataframe


def cltv_final(dataframe, segment_count, segments, csv=True):
    """
        Segments the customers based on CLTV and optionally saves the result to a CSV file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input DataFrame.
        segment_count : int
            Number of segments for customer segmentation.
        segments : list
            List of labels for customer segments.
        csv : bool, optional
            Whether to save the result to a CSV file, default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing customer segments based on CLTV.

    """
    dataframe = modelling(dataframe)
    dataframe["segment"] = pd.qcut(dataframe["cltv"], segment_count, segments)

    if csv:
        dataframe.to_csv("cltv_flo.csv")

    return dataframe


df_org = pd.read_csv("dataset/flo_data_20k.csv")
df = df_org.copy()

cltv_final(df, 6, ["F", "D", "C", "B", "A", "S"])
