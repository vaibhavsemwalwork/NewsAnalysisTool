from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from translate import Translator
from gtts import gTTS
import os
import google.generativeai as genai
import json
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# scrape article links
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import io

# extract content
import requests
from bs4 import BeautifulSoup

# sentiment analysis and article summary
from transformers import pipeline

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
from flask import Flask, request, Response
from flask import send_file
from flask_cors import CORS
import io

# Load API keys and model type
from config import GEMINI_API_KEY, MODEL_TYPE, n
api_key = GEMINI_API_KEY
model_type = MODEL_TYPE

from utils import scrape, extract_content, analyze_sentiment, summarize_article, analyze_articles, topic_of_text, comparative_analysis, get_summary, split_text, translate_text, text_to_speech_gtts, final_sentiment, report



app = Flask(__name__)
CORS(app)

# Gemini API setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_TYPE)



# API endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    company_name = data.get("company_name")

    # Scrape article links
    all_links = scrape(company_name)

    # extract content
    articles = extract_content(all_links[:n])
    print("\nNumber of articles : ",len(articles))

    # sentiment analysis
    sentiment, sentiment_score = analyze_sentiment(articles)
    print("\nSentiments analysis done. ",len(sentiment))
    print("\nSample :",sentiment[0])

    # Analyze articles and generate summaries
    analyzed_articles = analyze_articles(articles)
    print("\nArticle analyzed")
    print("\nSample : ",analyzed_articles[0])

    # topic extraction
    topics = topic_of_text(api_key, articles, model_type)
    print("\nTpoics Extracted")
    print("\nSample:", topics[0])

    # comparative analysis
    comparision = comparative_analysis(analyzed_articles)
    print("\nComparative analysis Done")

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
    fin_sentiment = final_sentiment(summary)
    print("\nfinal_Sentiment :",fin_sentiment)
    
    # creating Final report
    fin_report = report(company_name, sentiment, sentiment_score, analyzed_articles, topics, comparision, fin_sentiment, audio_file)

    
    return jsonify(fin_report)#, hindi



@app.route('/plot', methods=['POST'])
def plot():
    report = request.json
    sent = report["Comparative Sentiment Score"]["Sentiment Distribution"]
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
    return send_file(buf, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)