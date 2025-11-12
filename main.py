import config
from agents.master_agent import MasterAgent


def run():
    """
    Main entry point for the FinScope system.
    Coordinates data fetching, sentiment analysis, preprocessing,
    and LSTM-based stock price prediction using the MasterAgent.
    """

    print("\n===================================")
    print(f"     FinScope AI Prediction")
    print("===================================")
    print(f"Ticker: {config.TICKER}")
    print(f"Date Range: {config.START_DATE} → {config.END_DATE}")
    print("===================================\n")

    # --- Initialize Master Agent ---
    master_agent = MasterAgent(seq_len=3, use_sentiment=True)

    # --- Run Full Workflow ---
    try:
        predicted_price = master_agent.run(
            stock_symbol=config.TICKER,
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )

        if predicted_price is not None:
            print("\n===================================")
            print(f"Predicted Stock Price for {config.TICKER}: ${predicted_price['prediction']:.2f}")
            print("===================================\n")
        else:
            print("\nPrediction failed. Please check logs or model settings.\n")

    except Exception as e:
        print(f"\nAn unexpected error occurred during pipeline execution: {e}\n")


if __name__ == "__main__":
    """
    Run the full FinScope pipeline:
    1. Fetch stock + news data
    2. Analyze sentiment using Gemini
    3. Preprocess price + sentiment data
    4. Predict stock price with LSTM
    """
    run()
