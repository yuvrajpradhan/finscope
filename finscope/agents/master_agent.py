class MasterAgent:
    """
    The Master Agent that aggregates inputs and makes a final decision.
    """
    def decide(self, lstm_prediction, current_price, sentiment, event):
        """
        Makes a 'Buy', 'Sell', or 'Hold' decision based on agent inputs.
        
        Args:
            lstm_prediction (float): The predicted price from the LSTM agent.
            current_price (float): The actual current stock price.
            sentiment (str): The overall sentiment ('Positive', 'Negative', 'Neutral').
            event (dict): The event dictionary from the LLM agent.
            
        Returns:
            str: The final decision: "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell".
        """
        print("\n--- Master Agent Decision Logic ---")
        print(f"LSTM Prediction: {lstm_prediction:.2f}")
        print(f"Current Price:   {current_price:.2f}")
        print(f"Sentiment:       {sentiment}")
        print(f"Event Detected:  {event.get('event_type', 'none')}")
        
        # Start with a base score
        score = 0
        
        # --- Rule 1: Price Prediction ---
        price_diff_percent = (lstm_prediction - current_price) / current_price
        if price_diff_percent > 0.02: # Predicts >2% increase
            score += 2
        elif price_diff_percent > 0.005: # Predicts >0.5% increase
            score += 1
        elif price_diff_percent < -0.02: # Predicts >2% decrease
            score -= 2
        elif price_diff_percent < -0.005: # Predicts <0.5% decrease
            score -= 1
            
        # --- Rule 2: Sentiment ---
        if sentiment == 'Positive':
            score += 1
        elif sentiment == 'Negative':
            score -= 1
            
        # --- Rule 3: Events ---
        event_type = event.get('event_type', 'none')
        positive_events = ['product launch', 'analyst upgrade', 'merger & acquisition', 'earnings report']
        negative_events = ['analyst downgrade', 'legal issues']
        
        if event_type in positive_events:
            score += 2
        elif event_type in negative_events:
            score -= 2
            
        # --- Final Decision ---
        if score >= 3:
            return "Strong Buy"
        elif score == 2:
            return "Buy"
        elif score == -1:
            return "Sell"
        elif score <= -2:
            return "Strong Sell"
        else: # score is 0 or 1
            return "Hold"