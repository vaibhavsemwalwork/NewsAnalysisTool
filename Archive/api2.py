from quart import Quart, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from translate import Translator
from gtts import gTTS
import os
import google.generativeai as genai
import json
import matplotlib.pyplot as plt
import matplotlib
import io
import asyncio
from quart_cors import cors

import warnings
# Suppress asyncio warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")

# Load API keys and model type
from config import GEMINI_API_KEY, MODEL_TYPE, n
import logging
logging.basicConfig(level=logging.INFO)

from utils import (
    scrape, 
    extract_content, 
    analyze_sentiment, 
    analyze_articles,
    topic_of_text, 
    comparative_analysis, 
    get_summary, 
    translate_text, 
    text_to_speech_gtts, 
    final_sentiment, 
    report
)

app = Quart(__name__)
app = cors(app)

# Gemini API setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_TYPE)
api_key = GEMINI_API_KEY
model_type = MODEL_TYPE

# API endpoint
@app.route('/analyze', methods=['POST'])
async def analyze():
    data = await request.get_json()
    company_name = data.get("company_name")
    logging.info(f"Company name : {company_name}")

    # Scrape article links
    all_links = scrape(company_name)
    logging.info("Links Gathered")

    # extract content
    articles = await extract_content(all_links[:n])
    logging.info(f"\nSentiments analysis done. ")
    logging.info(f"\nNumber of articles : {len(articles)}")

    # sentiment analysis
    sentiment, sentiment_score = await analyze_sentiment(articles)
    
    "\nNumber of articles : ", len(articles)
    print("\nSentiments analysis done. ", len(sentiment))
    print("\nSample :", sentiment[0])

    # Analyze articles and generate summaries
    # analyzed_articles = analyze_articles(articles)
    analyzed_articles = await analyze_articles(articles)
    # analyzed_articles = asyncio.run(analyze_articles(articles))
    print("\nArticle analyzed")
    print("\nSample : ", analyzed_articles[0])

    # topic extraction
    topics = topic_of_text(api_key, articles, model_type)
    print("\nTopics Extracted")
    print("\nSample:", topics[0])

    # comparative analysis
    comparison = comparative_analysis(analyzed_articles)
    print("\nComparative analysis Done")

    # final summary
    summary = get_summary(company_name, analyzed_articles, sentiment_score)
    print("\n", summary)

    # Translate to Hindi
    hindi = translate_text(summary)
    print("\nHindi Text : ", hindi)

    # Convert text to speech
    audio_file = text_to_speech_gtts(hindi, language="hi", output_file=f"Audio/{company_name}.mp3")
    print("\npath Audio file : ", audio_file)

    # Final Sentiment Analysis
    fin_sentiment = final_sentiment(summary)
    print("\nfinal_Sentiment :", fin_sentiment)
    
    # creating Final report
    fin_report = report(company_name, sentiment, sentiment_score, analyzed_articles, topics, comparison, fin_sentiment, audio_file)

    return jsonify(fin_report)

@app.route('/plot', methods=['POST'])
async def plot():
    report_data = await request.get_json()
    sent = report_data["Comparative Sentiment Score"]["Sentiment Distribution"]
    sentiments = list(sent.keys())
    counts = list(sent.values())

    # Convert counts to percentages
    total = sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    # Create a bar plot
    fig, ax = plt.subplots()
    ax.bar(sentiments, percentages, color=['red', 'gray', 'green'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")
    ax.set_title("Sentiment Distribution (Percentage)")

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Return the image as a response
    return await send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)