CREATE TABLE raw_prices (
    "Date" DATE,
    "Asset" VARCHAR(20),
    "Close_Price" NUMERIC 
);
CREATE TABLE nifty_prices (
    "Date" DATE,
    "Nifty_Close" NUMERIC
);
copy nifty_prices FROM 'path/nifty50_prices.csv' WITH CSV HEADER;

copy raw_prices FROM 'path/raw_prices.csv' WITH CSV HEADER

CREATE OR REPLACE VIEW ml_feature_set AS
WITH LogReturns AS (
    SELECT 
        "Date",
        "Asset",
        "Close_Price",
        LN("Close_Price" / LAG("Close_Price") OVER (PARTITION BY "Asset" ORDER BY "Date")) AS log_return
    FROM raw_prices
)
SELECT 
    "Date",
    "Asset",
    log_return AS target_return,
    LAG(log_return, 1) OVER w AS lag_1,
    LAG(log_return, 2) OVER w AS lag_2,
    LAG(log_return, 5) OVER w AS lag_5,
    STDDEV(log_return) OVER (PARTITION BY "Asset" ORDER BY "Date" ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS rolling_vol_21d
FROM LogReturns
WHERE log_return IS NOT NULL
WINDOW w AS (PARTITION BY "Asset" ORDER BY "Date");


copy (SELECT * FROM ml_feature_set WHERE lag_5 IS NOT NULL AND rolling_vol_21d IS NOT NULL) TO 'path/portfolio_features.csv' WITH CSV HEADER;

