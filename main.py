import os
import requests
import csv
import nltk
from collections import Counter
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# List of stopwords (common words to exclude, e.g., adjectives like "Double")
stop_words = set(stopwords.words('english'))

# Set your API key (set this in your environment variables for security)
# API_KEY = os.getenv("NEWS_API_KEY")  # Replace with your own key for testing
API_KEY = "ff8d06ebb29b463793ade65ef092485a"

# API endpoint
API_ENDPOINT = "https://newsapi.org/v2/everything"

def search_news(topic, language="en"):
    """
    Search for news articles using the News API.
    :param topic: The search topic.
    :param language: Language code for filtering articles.
    :return: List of articles with titles, URLs, and publication dates.
    """
    params = {
        "q": topic,
        "language": language,
        "sortBy": "relevancy",
        "apiKey": API_KEY
    }
    try:
        response = requests.get(API_ENDPOINT, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        articles = data.get("articles", [])
        return [
            {
                "title": article["title"],
                "url": article["url"],
                "publishedAt": article["publishedAt"]
            }
            for article in articles
        ]
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        return []

def save_to_csv(articles, topic):
    """
    Save the list of articles to a CSV file.
    :param articles: List of articles to save.
    :param topic: The search topic (used for the file name).
    """
    filename = f"{topic.replace(' ', '_')}_articles.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Title", "URL", "Published At"])
        # Write the article data
        for article in articles:
            writer.writerow([article["title"], article["url"], article["publishedAt"]])
    print(f"\nResults saved to {filename}")

def summarize_with_sumy(headlines):
    """
    Summarize the given list of headlines using Sumy.
    :param headlines: List of headlines to summarize.
    :return: A summary of the headlines.
    """
    text = " ".join(headlines)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize into 3 sentences
    return " ".join(str(sentence) for sentence in summary)

def extract_named_entities(headlines):
    """
    Extract and count named entities from the headlines using NLTK.
    :param headlines: List of article headlines.
    :return: Sorted list of named entities with their frequencies.
    """
    entity_counter = Counter()  # To count the frequency of named entities
    
    for headline in headlines:
        # Tokenize the headline into words and perform POS tagging
        tokens = word_tokenize(headline)
        tagged_tokens = pos_tag(tokens)
        
        # Apply NLTK's named entity chunker
        tree = ne_chunk(tagged_tokens)
        
        # Extract named entities from the tree
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                # If the subtree is a named entity, extract the entity name and label
                entity = " ".join(word for word, tag in subtree)
                label = subtree.label()
                if label in ['GPE', 'PERSON', 'ORG']:  # Only interested in persons, locations, and organizations
                    entity_counter[entity] += 1
    
    # Return the named entities sorted by frequency
    sorted_entities = entity_counter.most_common()
    return sorted_entities

def main():
    while True:
        print("Welcome to the News Search Tool!")
        topic = input("Enter the topic you want to search for (or type 'exit' to quit): ").strip()
        if topic.lower() == "exit":
            print("Exiting the application. Goodbye!")
            break
        
        language = input("Enter the language code (default: 'en'): ").strip() or "en"
        
        print(f"\nSearching for articles about '{topic}' in '{language}'...")
        articles = search_news(topic, language)
        
        if articles:
            # Display and save the top 15 articles
            top_articles = articles[:15]
            print("\nTop Articles:")
            top_headlines = []
            for idx, article in enumerate(top_articles, 1):
                print(f"{idx}. {article['title']} ({article['publishedAt']})")
                print(f"   URL: {article['url']}")
                top_headlines.append(article["title"])
            
            save_to_csv(top_articles, topic)
            
            print("\nGenerating summary of top headlines...")
            summary = summarize_with_sumy(top_headlines)
            print("\nSummary of Top Headlines:")
            print(summary)
            
            print("\nIdentifying and counting named entities in the headlines...")
            entity_counts = extract_named_entities(top_headlines)
            print(entity_counts)
        else:
            print("No articles found.")
            
        # Ask the user if they want to search again
        continue_search = input("\nDo you want to search again? (yes/no): ").strip().lower()
        if continue_search != "yes":
            print("Exiting the application. Goodbye!")
            break

if __name__ == "__main__":
    main()
