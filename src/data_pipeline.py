import pandas as pd
from .config import RAW_DIR, PROCESSED_DIR

EXPECTED = {
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
}

def find_excel_file():
    files = list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls"))
    if not files:
        raise FileNotFoundError(
            "No Excel dataset found. Put the UCI Online Retail II file in data/raw/."
        )
    return files[0]

def load_raw_data():
    path = find_excel_file()
    print(f"Loading: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    missing = EXPECTED - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print(f"Raw shape: {df.shape}")
    return df

def clean_transactions(df):
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(
        subset=["InvoiceDate", "Customer ID", "Quantity", "Price"]
    )

    df["Invoice"] = df["Invoice"].astype(str)
    df["IsCancellation"] = df["Invoice"].str.startswith("C")

    # Keep completed sales for customer behavior modeling.
    df = df[~df["IsCancellation"]]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df.drop_duplicates()

    df["Revenue"] = df["Quantity"] * df["Price"]

    # Normalize mixed-type categorical columns before Parquet export
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    output = PROCESSED_DIR / "clean_transactions.parquet"
    df.to_parquet(output, index=False)

    print(f"Clean shape: {df.shape}")
    print(f"Customers: {df['Customer ID'].nunique()}")
    print(f"Saved: {output}")

    return df
