import json
import requests
import pandas as pd
import nltk
import re
import string
from datetime import date, timedelta
from typing import List, Dict
from dataclasses import dataclass
from transformers import pipeline
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords')

@dataclass
class NewsAPIConfig:
    """Configuration for NewsAPI requests"""
    base_url: str = 'https://newsapi.org/v2/everything?'
    api_key: str = 'fa422c784ca843a0bb09c0a9381a4abf'
    sort_by: str = 'popularity'
    
    def get_date_range(self, days: int = 30) -> str:
        """Generate date range string for API query"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        return f'from={start_date}&to={end_date}&'

class NewsAnalyzer:
    def __init__(self, config: NewsAPIConfig):
        self.config = config
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.stop_words = set(stopwords.words('english'))

    def fetch_news(self, topics: List[str]) -> pd.DataFrame:
        """Fetch news articles for given topics"""
        news_data = []
        date_range = self.config.get_date_range()
        
        for topic in topics:
            query_url = (
                f"{self.config.base_url}"
                f"q={topic}&"
                f"{date_range}"
                f"sortBy={self.config.sort_by}"
                f"&apiKey={self.config.api_key}"
            )
            response = requests.get(query_url).json()
            news_data.append(response)
        
        return pd.json_normalize(news_data, record_path='articles')

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing special characters, numbers, and converting to lowercase"""
        text = text.lower()
        text = re.sub('\[.*?\]', "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub("\w*\d\w*", "", text)
        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        return ' '.join([word for word in text.split() if word not in self.stop_words])

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of given text"""
        result = self.sentiment_pipeline(text)[0]
        return {
            'sentiment': result['label'],
            'score': result['score']
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe with text cleaning and sentiment analysis"""
        # Clean text and remove stopwords
        df['cleaned_text'] = df['description'].apply(self.clean_text)
        df['cleaned_final'] = df['cleaned_text'].apply(self.remove_stopwords)
        
        # Analyze sentiment
        sentiments = df['description'].apply(self.analyze_sentiment)
        df['sentiment'] = sentiments.apply(lambda x: x['sentiment'])
        df['score'] = sentiments.apply(lambda x: x['score'])
        
        # Additional processing
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['description_length'] = df['description'].str.split().str.len()
        df['author'] = df['author'].fillna('author not indicated')
        
        # Clean up intermediate columns
        df = df.drop(['cleaned_text', 'source.id'], axis=1)
        
        return df

def main():
    # Initialize configuration
    config = NewsAPIConfig()
    analyzer = NewsAnalyzer(config)
    
    # Define topics
    financial_topics = ['bitcoin', 'finance', 'metaverse', 'recession']
    news_topics = ['war', 'election', 'covid', 'Trump']
    
    # Fetch and process news
    financial_df = analyzer.fetch_news(financial_topics)
    news_df = analyzer.fetch_news(news_topics)
    
    # Combine and process final dataset
    final_df = pd.concat([financial_df, news_df], ignore_index=True)
    final_df = analyzer.process_dataframe(final_df)
    
    # Save to CSV
    final_df.to_csv("news_analysis.csv", index=False)
    
    return final_df

if __name__ == "__main__":
    df_final = main()
