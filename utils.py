# scrape article links
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# extract content
import requests
from bs4 import BeautifulSoup

# sentiment analysis
from transformers import pipeline

# article summary
from transformers import pipeline
# import torch

# topic extract
import google.generativeai as genai

# comparative analysis
import google.generativeai as genai
import json

# translate to hindi
from translate import Translator

# TTS
from gtts import gTTS
import os

import aiohttp
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
logging.basicConfig(level=logging.INFO)

from config import GEMINI_API_KEY, MODEL_TYPE, n
api_key = GEMINI_API_KEY
model_type = MODEL_TYPE


# ----------------------------------------scrape article links------------------------------------------------------------------ #
def scrape(company_name):
    # Set up Chrome WebDriver
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    # Function to scrape links from a given page URL
    def scrape_links(url):
        driver.get(url)
        # Wait for the page to load (adjust the sleep time if needed)
        time.sleep(5)

        # Find all divs with the specified attributes
        cards = driver.find_elements(By.XPATH, '//div[@data-testid="newport-card" and @data-indexcard="true"]')

        # Initialize a list to store the extracted data
        news_data = []

        # Loop through the cards and extract the required information
        for card in cards:
            try:
                # Find the nested div with data-testid="anchor-inner-wrapper"
                anchor_wrapper = card.find_element(By.XPATH, './/div[@data-testid="anchor-inner-wrapper"]')
                
                if anchor_wrapper:
                    # Find the <a> tag with data-testid="internal-link"
                    link_element = anchor_wrapper.find_element(By.XPATH, './/a[@data-testid="internal-link"]')
                    
                    if link_element:
                        # Extract the link
                        link = link_element.get_attribute("href")
                    else:
                        link = "N/A"
                else:
                    link = "N/A"
                
                # Append the extracted data to the list
                news_data.append({
                    "link": link
                })
            except Exception as e:
                # Handle any exceptions (e.g., missing elements)
                print(f"Error processing card: {e}")
        
        return news_data



    # Scrape links from page 1
    url_page1 = f"https://www.bbc.com/search?q={company_name}&page=0"
    print("Scraping page 1...")
    page1_data = scrape_links(url_page1)
    print("\nLinks from page 1:\n", page1_data)

    # Scrape links from page 2
    url_page2 = f"https://www.bbc.com/search?q={company_name}&page=1"
    print("Scraping page 2...")
    page2_data = scrape_links(url_page2)
    print("\nLinks from page 2:\n", page2_data)

    # Combine data from both pages
    all_links = page1_data + page2_data
    print("\nAll links:\n", all_links)

    # Close the browser
    driver.quit()

    print(f"Searched BBC for '{company_name}' and closed successfully.")

    return all_links


# --------------------------------------extract title and content from an article link------------------------------------------- #
async def extract_article(session, article_url):
    try:
        # Fetch the article page
        async with session.get(article_url) as response:
            response.raise_for_status()  # Raise an error for bad status codes

            # Parse the HTML content using BeautifulSoup
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

            # Extract the title
            title_element = soup.find("h1")
            title = title_element.text.strip() if title_element else "N/A"

            # Extract the content
            content_elements = soup.find_all("p")  # Assuming content is in <p> tags
            content = "\n".join([p.text.strip() for p in content_elements]) if content_elements else "N/A"

            return [title, content]
    except Exception as e:
        print(f"Error extracting article from {article_url}: {e}")
        return {
            "title": "N/A",
            "content": "N/A"
        }

async def extract_content(article_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [extract_article(session, link['link']) for link in article_urls]
        articles = await asyncio.gather(*tasks)
        return articles


# ------------------------------------------sentiment analysis-------------------------------------------------------------------- #
async def analyze_sentiment(articles):
    """
    Perform sentiment analysis on a list of articles asynchronously.
    Each article is in the format ["title", "content"].
    Returns a list of dictionaries with sentiment results.
    """
    # Load the sentiment-analysis pipeline
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    # Initialize a list to store results
    sentiment_results = []

    # Function to analyze sentiment for a single article
    def analyze_article(article):
        title, content = article[0], article[1]
        try:
            # Split the content into chunks of <= 512 tokens
            chunk_size = 500  # Leave some room for special tokens
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

            # Analyze sentiment for each chunk
            chunk_sentiments = []
            chunk_confidences = []
            for chunk in chunks:
                result = sentiment_pipeline(chunk)[0]
                chunk_sentiments.append(result["label"])
                chunk_confidences.append(result["score"])

            # Aggregate the results (e.g., take the majority sentiment)
            sentiment = max(set(chunk_sentiments), key=chunk_sentiments.count)
            confidence = sum(chunk_confidences) / len(chunk_confidences)  # Average confidence
        except Exception as e:
            # Handle errors (e.g., if the content is too short)
            sentiment = "neutral"
            confidence = 0.0

        return {
            "title": title,
            "sentiment": sentiment,
            "confidence": confidence
        }

    # Run sentiment analysis for all articles concurrently
    tasks = [asyncio.to_thread(analyze_article, article) for article in articles]
    sentiment_results = await asyncio.gather(*tasks)

    # Generate comprehensive report
    def sentiment_score(sentiment):
        positive, negative, neutral = 0, 0, 0

        for i in sentiment:
            if i['sentiment'] == 'POSITIVE':
                positive += 1
            elif i['sentiment'] == 'NEGATIVE':
                negative += 1
            else:
                neutral += 1

        result = {
            'POSITIVE': positive,
            'NEGATIVE': negative,
            'NEUTRAL': neutral
        }

        return result

    sentiment_score_result = sentiment_score(sentiment_results)

    return sentiment_results, sentiment_score_result


# -----------------------------------------------article analysis--------------------------------------------------------------------------- #
semaphore = asyncio.Semaphore(5)

async def summarize_article(summarizer, content, max_length=50, max_input_length=1024):
    """
    Generate a single-line summary of an article using DistilBART.
    """
    
    chunks = [content[i:i + max_input_length] for i in range(0, len(content), max_input_length)]

    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=10, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Error summarizing chunk: {e}")
            summaries.append("Unable to generate summary for this chunk.")

    combined_summary = " ".join(summaries)
    return combined_summary

async def analyze_article(summarizer, article):
    """
    Analyze a single article and return the result.
    """
    

    async with semaphore:  # Limit to 3 concurrent tasks
        title, content = article[0], article[1]

        # Generate a single-line summary
        summary = await summarize_article(summarizer, content)

        return {
            "title": title,
            "summary": summary
        }

async def analyze_articles(articles):
    """
    Analyze a list of articles and generate summaries concurrently,
    with at most 3 articles being analyzed at the same time.
    Each article is in the format ["title", "content"].
    Returns a list of dictionaries with analysis results.
    """
    logging.info("Loading summarizer pipeline...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    tasks = [analyze_article(summarizer, article) for article in articles]
    results = await asyncio.gather(*tasks)
    return results



# ------------------------------------------topic extraction----------------------------------------------------------------------- #
def topic_of_text(api_key, articles, model='gemini-2.0-flash', max_output_tokens=200):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model)
    topics = []

    for article in articles:
        _, content = article[0], article[1]
        prompt = f"""
        Extract five topics (one or two words per topic only) discussed in the following text. Format the output exactly as a Python list of strings, like this: ["topic 1", "topic 2", "topic 3", "topic 4", "topic 5"]. 
        
        - Ensure the output is a valid Python list representation with double quotes around each topic.
        - Do not include any additional formatting, explanations, or code block markers like ```python or ```.

        Text:
        {content}

        Topics:
        """

        response = model.generate_content(prompt, generation_config={"max_output_tokens": max_output_tokens})
        topics.append(response.text.strip())
    return topics



# ------------------------------------------comparative analysis------------------------------------------------------------------- #
def comparative_analysis(analyzed_articles):
    # Configure the Gemini API
    genai.configure(api_key=api_key)  # Replace with your actual API key

    # Initialize the Gemini Flash 2.0 model
    model = genai.GenerativeModel(MODEL_TYPE)

    # Define the articles
    articles = [analyzed_articles[i]['summary'] for i in range(len(analyzed_articles))]

    # Define the prompt for the Gemini API
    prompt = """
    Perform a comparative analysis of the following articles. The output should be a dictionary in JSON format with two keys.:
    1. "Coverage Differences": A list of dictionaries, each containing "Comparison" and "Impact" keys.
    2. "Topic Overlap": A dictionary with "Common Topics", "Unique Topics in Article 1", "Unique Topics in Article 2", "Unique Topics in Article 3", and so on.
    Ouput be such that it can be directly added to json. Don't provide any explanation
    Here are the articles:
    """ + "\n".join([f"Article {i+1}: {article}" for i, article in enumerate(articles)])

    # Generate the response using Gemini Flash 2.0
    response = model.generate_content(prompt)

    # Extract the response text
    response_text = response.text

    # Print the raw response for debugging
    # print("Raw Response from Gemini:")
    # print(response_text)

    # Remove Markdown code block markers (```json and ```)
    if response_text.startswith("```json") and response_text.endswith("```"):
        response_text = response_text[7:-3].strip()  # Remove ```json and ```

    # Try to parse the response as JSON
    try:
        analysis_dict = json.loads(response_text)
        # print("\nParsed Dictionary:")
        # print(json.dumps(analysis_dict, indent=4))  # Pretty-print the dictionary
    except json.JSONDecodeError as e:
        print("\nError parsing JSON. The response is not in valid JSON format.")
        print(f"Error: {e}")
    
    return analysis_dict


# ----------------------------------------------summary---------------------------------------------------------------------------- #
def get_summary(company_name, analyzed_articles, sentiment_score):
    # Configure the Gemini API
    genai.configure(api_key=api_key)  # Replace with your actual API key

    # Initialize the Gemini Flash 2.0 model
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Define the articles
    articles = [analyzed_articles[i]['summary'] for i in range(len(analyzed_articles))]

    prompt = f"""
    You are an expert analyst summarizing multiple news articles related to the company '{company_name}'. Your task is to create a comprehensive summary that includes the following:

    1. **Sentiment Overview**: Begin the summary with a sentence that describes the overall sentiment of the articles based on the following distribution: {sentiment_score}. For example, "The articles have a predominantly negative sentiment, with 13 negative articles, 5 positive articles, and 0 neutral articles."
    2. **Key Details**: Extract the most important information from the articles, such as events, announcements, or developments related to the company.
    3. **Sentiment Analysis**: Analyze the overall sentiment (positive, negative, or neutral) expressed in the articles toward the company. Provide a brief explanation for the sentiment.
    4. **Comparative Analysis**: Highlight any trends, patterns, or contrasting viewpoints across the articles. For example, are there conflicting opinions or consistent themes?
    5. **Conciseness**: Ensure the summary is concise and well-structured, limited to 150 words, so it can be easily converted into a Hindi text-to-speech output.

    NOTE : Make the first sentence about the sentiment in this {sentiment_score}.
    Here are the articles:
    """ + "\n".join([f"Article {i+1}: {article}" for i, article in enumerate(articles)]) + """

    """

    # Generate the response using Gemini Flash 2.0
    response = model.generate_content(prompt)

    # Extract the response text
    response_text = response.text

    return response_text


# -------------------------------------------translate hindi----------------------------------------------------------------------- #
# Function to split text into chunks of a specified size
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Translate English to Hindi in chunks
def translate_text(text, chunk_size=500):
    translator = Translator(from_lang="en", to_lang="hi")
    chunks = split_text(text, chunk_size)
    translated_chunks = []
    
    for chunk in chunks:
        translated = translator.translate(chunk)
        translated_chunks.append(translated)
    
    # Combine the translated chunks
    return " ".join(translated_chunks)

# ---------------------------------------------TTS (speech)------------------------------------------------------------------------- #
def text_to_speech_gtts(text, language="hi", output_file="Audio/output.mp3"):
    """
    Convert text to Hindi speech using gTTS.
    
    Args:
        text (str): The text to convert to speech.
        language (str): Language code (e.g., "hi" for Hindi).
        output_file (str): Name of the output audio file.
    
    Returns:
        str: Path to the generated audio file.
    """
    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang=language, slow=False)

        # Make directory
        os.makedirs("Audio", exist_ok=True)

        # Save the audio file
        tts.save(output_file)

        return output_file
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None


# ---------------------------------------------Final Sentiment----------------------------------------------------------------------- #
def final_sentiment(summary):
    # Configure the Gemini API
    genai.configure(api_key=api_key)  # Replace with your actual API key

    # Initialize the Gemini Flash 2.0 model
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    From the given summary of many articles. Give final verdict on sentiment analysis in a one or two lines in very comprehensive way, with a point that needs to be highlight about the company based on summary(optional).

    Here is the summary :
    {summary}

    """

    # Generate the response using Gemini Flash 2.0
    response = model.generate_content(prompt)

    # Extract the response text
    response_text = response.text

    return response_text


# -----------------------------------------Plot sentiment distribution--------------------------------------------------------------- #
def plot_sentiment(report):
    sent = report["Comparative Sentiment Score"]["Sentiment Distribution"]
    sentiments = list(sent.keys())
    counts = list(sent.values())

    # Convert counts to percentages
    total = sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    # Create a bar plot
    plt.figsize
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.bar(sentiments, percentages, color=['red', 'gray', 'green'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")
    ax.set_title("Sentiment Distribution")

    return fig


# --------------------------------------------------Report--------------------------------------------------------------------------- #
def report(company_name, sentiment, sentiment_score, analyzed_articles, topics, comparision, final_sentiment, audio_file):

    output = {
        "Company" : company_name,
        "Articles" : [
            {"Title" : analyzed_articles[i]['title'], "Summary" : analyzed_articles[i]['summary'], "Sentiment" : sentiment[i]['sentiment'], "Topics" : topics[i]}
        for i in range(len(analyzed_articles))],
        "Comparative Sentiment Score" : {
            "Sentiment Distribution": sentiment_score,
            "Coverage Differences": comparision['Coverage Differences'],
            "Topic Overlap": comparision['Topic Overlap']

        } ,
        "Final Sentiment Analysis": final_sentiment,
        "Audio": audio_file
    }

    # Make directory
    os.makedirs("JSON", exist_ok=True)

    # Save the output dictionary as a JSON file
    with open(f"JSON/{company_name}", "w", encoding="utf-8") as json_file:
        json.dump(output, json_file, ensure_ascii=False, indent=4)

    return output


if __name__ == "__main__":
    # Define the company name
    company_name = "OpenAi"

    # Scrape article links
    all_links = scrape(company_name)
    print("\nlen of links : ",len(all_links))

    # extract content
    articles = asyncio.run(extract_content(all_links[:n]))
    print("\nlength of articles : ",len(articles))

    # sentiment analysis
    sentiment, sentiment_score = asyncio.run(analyze_sentiment(articles))
    print("\nLength of sentiments:", len(sentiment))
    print("\n", sentiment[0])

    # Analyze articles and generate summaries
    # analyzed_articles = analyze_articles(articles)
    analyzed_articles = asyncio.run(analyze_articles(articles))
    print("\nLength of analyzed articles:", len(analyzed_articles))
    print("\n", analyzed_articles[0])
    

    # topic extraction
    topics = topic_of_text(api_key, articles, model_type)
    print("\nlength of topics : ",len(topics))
    print("\ntopic:", topics)

    # comparative analysis
    comparision = comparative_analysis(analyzed_articles)
    print("\ncomparision : ",comparision)

    # final summary
    summary = get_summary(company_name, analyzed_articles, sentiment_score)
    print("\n",summary)

    # Translate to Hindi
    hindi = translate_text(summary)
    print("\nHindi Text : ",hindi)

    # Convert text to speech
    audio_file = text_to_speech_gtts(hindi, language="hi", output_file=f"Audio/{company_name}.mp3")
    print("\npath Audio file : ",audio_file)

    # Final Semtiment Analysis
    final_sentiment = final_sentiment(summary)
    print("\nfinal_Sentiment :",final_sentiment)

    # creating Final report
    report = report(company_name, sentiment, sentiment_score, analyzed_articles, topics, comparision, final_sentiment, audio_file)
    print("\nreport :",report)
