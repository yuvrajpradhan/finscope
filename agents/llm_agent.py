import os
import json
import requests
import config

class LLMAgent:
    """
    LLM Agent using Gemini REST API (v1).
    Works with gemini-1.5-flash, gemini-1.5-pro, gemini-1.5-flash-latest, etc.
    """

    def __init__(self):
        self.api_key = config.API_KEY_GEMINI
        if not self.api_key:
            raise ValueError("Gemini API key missing in config.py")

        self.model = "gemini-2.5-flash"
        self.url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"

        print(f"LLMAgent initialized with model: {self.model}")

    def analyze_sentiment(self, text):
        print(f"Analyzing sentiment for: '{text[:50]}...'")

        prompt = f"""
        Analyze the sentiment of this financial headline.
        Return only one word: Positive, Negative, or Neutral.

        Headline: "{text}"
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            response = requests.post(self.url, json=payload, timeout=20)
            data = response.json()

            sentiment = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if sentiment not in ["Positive", "Negative", "Neutral"]:
                return "Neutral"

            return sentiment

        except Exception as e:
            print("Sentiment analysis error:", e)
            return "Neutral"

    def analyze_daily_batch_sentiment(self, daily_headlines_dict: dict):
        # Convert the input Python dict into a JSON string for the prompt
        input_dict_string = json.dumps(daily_headlines_dict, indent=2) 
        
        prompt = f"""
        TASK: For each date provided, analyze the combined sentiment of all corresponding news headlines.
        SCORING RULES: Calculate a single net sentiment score for each date ranging from -1.0 (bearish) to +1.0 (bullish).

        INPUT DATA:
        {input_dict_string}

        OUTPUT FORMAT:
        Respond STRICTLY in JSON. The output must be a single dictionary where keys are the dates (YYYY-MM-DD) and values are the calculated net sentiment scores (float, e.g., 0.45).
        DO NOT include any explanation or extra text. Your response must start with '{' and end with '}' and contain nothing else.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
            # "generationConfig": {"responseMimeType": "application/json"}
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=120) 
            data = response.json()
            
            # Extract and parse the JSON response text
            json_text = data["candidates"][0]["content"]["parts"][0]["text"]

            if json_text.startswith("```json"):
                json_text = json_text.replace("```json", "").replace("```", "").strip()
            
            # The output is directly the dictionary you wanted!
            return json.loads(json_text) 

        except Exception as e:
            print("Batch sentiment dictionary analysis error:", e)
            return {} # Return an empty dict on failure

    def extract_events(self, text):
        print(f"Extracting events from: '{text[:50]}...'")

        prompt = f"""
        Identify any financial events in the text:
        - earnings report
        - merger & acquisition
        - product launch
        - analyst upgrade
        - analyst downgrade
        - legal issues

        Respond strictly in JSON:
        {{
            "event_detected": true/false,
            "event_type": "string"
        }}

        Text: "{text}"
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            response = requests.post(self.url, json=payload, timeout=20)
            data = response.json()

            json_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_text)

        except Exception as e:
            print("Event extraction error:", e)
            return {"event_detected": False, "event_type": "none"}
