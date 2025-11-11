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

        self.model = "gemini-1.5-flash"
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
