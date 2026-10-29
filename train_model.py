"""Re-train the built-in model on fresh data and save it.

    python train_model.py                # default ticker GC=F (gold futures, like the notebook data)
    python train_model.py --ticker AAPL

Writes model.pkl (scikit-learn Ridge) and model_params.json (plain coefficients).
"""
import argparse
import json

import joblib
from sklearn.linear_model import Ridge

import utils


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()

    df = utils.fetch_history(args.ticker, years=args.years)
    model, metrics = utils.train_model(df)
    print(f"Trained on {len(df)} days of {args.ticker}")
    print("Chronological test metrics:", json.dumps(metrics, indent=2))

    ridge = Ridge(alpha=1.0).fit(df[utils.FEATURES].to_numpy(float), df["Close"].to_numpy(float))
    joblib.dump(ridge, utils.MODEL_PATH)
    utils.PARAMS_PATH.write_text(json.dumps(model.to_dict(), indent=2))
    print(f"Saved {utils.MODEL_PATH.name} and {utils.PARAMS_PATH.name}")


if __name__ == "__main__":
    main()
