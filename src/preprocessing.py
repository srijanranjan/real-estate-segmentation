import numpy as np
import pandas as pd

RAW_CLIENTS_PATH = "data/clients.csv"
RAW_PROPERTIES_PATH = "data/properties.csv"

def clean_clients(path: str = RAW_CLIENTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    cat_cols = ["client_type", "gender", "country", "region",
                "acquisition_purpose", "loan_applied", "referral_channel"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="client_id", keep="first")

    def parse_dob(val: str):
        val = str(val).strip()
        if "-" in val:
            d, m, y = val.split("-")          # dash format = DD-MM-YYYY
        elif "/" in val:
            m, d, y = val.split("/")          # slash format = MM/DD/YYYY
        else:
            return pd.NaT
        try:
            return pd.Timestamp(year=int(y), month=int(m), day=int(d))
        except (ValueError, TypeError):
            return pd.NaT

    df["dob_parsed"] = df["date_of_birth"].apply(parse_dob)
    reference_date = pd.Timestamp("2025-12-31")
    df["age"] = ((reference_date - df["dob_parsed"]).dt.days / 365.25).round().astype("Int64")

    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median())

    df["loan_applied_flag"] = (df["loan_applied"] == "Yes").astype(int)
    df["is_company"] = (df["client_type"] == "Company").astype(int)
    df["is_investment_purpose"] = (df["acquisition_purpose"] == "Investment").astype(int)

    keep_cols = ["client_id", "client_type", "first_name", "last_name", "age", "gender",
                 "country", "region", "acquisition_purpose", "satisfaction_score",
                 "loan_applied", "loan_applied_flag", "referral_channel",
                 "is_company", "is_investment_purpose"]
    return df[keep_cols]


def clean_properties(path: str = RAW_PROPERTIES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["unit_category"] = df["unit_category"].astype(str).str.strip()
    df["listing_status"] = df["listing_status"].astype(str).str.strip()

    df["sale_price_clean"] = (
        df["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )

    df["transaction_date_parsed"] = pd.to_datetime(
        df["transaction_date"], format="%m-%d-%Y", errors="coerce"
    )

    df["price_per_sqft"] = df["sale_price_clean"] / df["floor_area_sqft"]
    df = df.drop_duplicates(subset="listing_id")
    return df


def build_client_features(clients_path=RAW_CLIENTS_PATH, properties_path=RAW_PROPERTIES_PATH):
    clients = clean_clients(clients_path)
    props = clean_properties(properties_path)

    sold = props[(props["listing_status"] == "Sold") & (props["client_ref"].notna())].copy()

    agg = sold.groupby("client_ref").agg(
        num_properties=("listing_id", "count"),
        total_spend=("sale_price_clean", "sum"),
        avg_ticket_size=("sale_price_clean", "mean"),
        max_ticket_size=("sale_price_clean", "max"),
        avg_floor_area=("floor_area_sqft", "mean"),
        total_floor_area=("floor_area_sqft", "sum"),
        avg_price_per_sqft=("price_per_sqft", "mean"),
        num_towers=("tower_number", "nunique"),
        first_purchase=("transaction_date_parsed", "min"),
        last_purchase=("transaction_date_parsed", "max"),
        pct_office=("unit_category", lambda s: (s == "Office").mean()),
    ).reset_index().rename(columns={"client_ref": "client_id"})

    agg["investment_tenure_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days

    agg["tenure_per_property"] = agg["investment_tenure_days"] / agg["num_properties"]

    agg["is_repeat_buyer"] = (agg["num_properties"] > 1).astype(int)

    merged = clients.merge(agg, on="client_id", how="left")

    fill_zero = ["num_properties", "total_spend", "avg_ticket_size", "max_ticket_size",
                 "avg_floor_area", "total_floor_area", "avg_price_per_sqft",
                 "num_towers", "investment_tenure_days", "tenure_per_property",
                 "is_repeat_buyer", "pct_office"]
    for col in fill_zero:
        merged[col] = merged[col].fillna(0)

    return merged


if __name__ == "__main__":
    features = build_client_features()
    features.to_csv("outputs/client_features.csv", index=False)
    print(f"Built client-level feature table: {features.shape}")
    print(features[["num_properties", "investment_tenure_days", "tenure_per_property"]].describe())