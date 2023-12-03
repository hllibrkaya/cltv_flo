import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def dataset_summary(dataframe):
    """
    This function visually prints basic summary statistics of a given pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame  to examine.
    Returns
    -------
    None
        The function only prints the outputs to the console and doesn't return any value.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### HEAD #####################")
    print(dataframe.head())
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### NULL VALUES #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
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
    quantile1 = dataframe[column].quantile(low_quantile)
    quantile3 = dataframe[column].quantile(up_quantile)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, column):
    low_limit, up_limit = outlier_thresholds(dataframe, column)
    if dataframe[(dataframe[column] > up_limit) | (dataframe[column] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_thresholds(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = round(low_limit, 0)
    dataframe.loc[(dataframe[column] > up_limit), column] = round(up_limit, 0)


data_org = pd.read_csv("dataset/flo_data_20k.csv")
df = data_org.copy()

dataset_summary(df)

categoric, _, numeric = grab_col_names(df)

for col in numeric:
    print(f"Col: {col} - {check_outlier(df, col)}")

for col in numeric:
    replace_with_thresholds(df, col)

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df[
    "customer_value_total_ever_online"]

date_cols = df.columns[df.columns.str.contains("date")]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

last_date = df["last_order_date"].max()

analysis_date = last_date + pd.Timedelta(days=2)

# recency: Time elapsed since the last purchase. Weekly.
# T: Customer's age. Weekly. (how long ago the first purchase was made from the analysis date)
# frequency: Total number of recurring purchases
# monetary: Average earnings per purchase

cltv = pd.DataFrame()
cltv["customer_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
cltv["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7
cltv["frequency"] = df["total_order_num"]
cltv["monetary_cltv_avg"] = df["customer_value_total"] / cltv["frequency"]

dataset_summary(cltv)

bgf = BetaGeoFitter(penalizer_coef=0.001)

plot_period_transactions(bgf)
plt.show()

bgf.fit(cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

cltv["exp_sales_3_month"] = bgf.predict(12, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])
cltv["exp_sales_6_month"] = bgf.predict(24, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"])

# customer life time value for 6 months
cltv["cltv"] = ggf.customer_lifetime_value(bgf, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"],
                                           cltv["monetary_cltv_avg"], time=6, freq="W", discount_rate=0.01)

cltv.sort_values(by="cltv", ascending=False)[:10]

cltv["segment"] = pd.qcut(cltv["cltv"], 4, ["D", "C", "B", "A"])

dataset_summary(cltv)

# observing 1 more segment
cltv.groupby("segment").agg({"cltv": ["mean", "max", "min"],
                             "frequency": ["mean", "max", "min"],
                             "exp_sales_6_month": ["mean", "max", "min", "count"],
                             "exp_average_value": ["mean", "max", "min"]
                             })

cltv["segment"] = pd.qcut(cltv["cltv"], 5, ["D", "C", "B", "A", "S"])

# just 1 more too...

cltv["segment"] = pd.qcut(cltv["cltv"], 6, ["F", "D", "C", "B", "A", "S"])
