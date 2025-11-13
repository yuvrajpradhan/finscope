import google.generativeai as genai
import config
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

class LLMAgent:
    """
    The LLM Agent for qualitative analysis using Google's Gemini.
    """
    def __init__(self):
        if not config.API_KEY_GEMINI or config.API_KEY_GEMINI == "YOUR_GEMINI_API_KEY":
            raise ValueError("Gemini API key not found in config.py. Please add it.")
        
        genai.configure(api_key=config.API_KEY_GEMINI)
        self.model = genai.GenerativeModel('gemini-pro')
        print("LLM Agent initialized with Gemini-Pro.")

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text (e.g., a news headline).
        
        Returns:
            str: 'Positive', 'Negative', or 'Neutral'.
        """
        print(f"Analyzing sentiment for: '{text[:50]}...'")
        prompt = f"""
        Analyze the sentiment of the following financial news headline.
        Classify it as 'Positive', 'Negative', or 'Neutral'.
        Return only the classification word.

        Headline: "{text}"
        Sentiment:
        """
        try:
            response = self.model.generate_content(prompt)
            sentiment = response.text.strip()
            if sentiment in ['Positive', 'Negative', 'Neutral']:
                return sentiment
            return 'Neutral' # Default if parsing fails
        except Exception as e:
            print(f"An error occurred during sentiment analysis: {e}")
            return "Neutral" # Default on error

    def extract_events(self, text):
        """
        Extracts key financial events from a text using structured output.
        
        Returns:
            dict: A dictionary containing extracted event info, or an empty dict.
        """
        print(f"Extracting events from: '{text[:50]}...'")
        prompt = f"""
        From the following financial news text, identify if any of these key events are mentioned: 
        'earnings report', 'merger & acquisition', 'product launch', 'analyst upgrade', 'analyst downgrade', 'legal issues'.
        
        Respond with a JSON object with two keys:
        1. "event_detected": boolean (true if an event was found, otherwise false).
        2. "event_type": string (the type of event found, or "none").

        Text: "{text}"
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"An error occurred during event extraction: {e}")
            return {"event_detected": False, "event_type": "none"}
