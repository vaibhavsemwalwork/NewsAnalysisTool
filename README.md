# NewsAnalysisTool
This tool extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi.

## Requirements
- Python 3.8+
- Streamlit
- Requests
- BeautifulSoup4
- Transformers
- Google GenerativeAI
- Translate
- gTTS
- Selenium
- flask
- gemini-2-flash api (free generate from google AI studio)

## Installation
1. Clone the repository.
2. (optional) Create python virtual environment ```python -m venv <environment name>``` and activate the environment ```source <\environment name>/bin/activate```
3. Install the required packages:
   ```pip install -r requirements.txt```
4. Create a config.py whith same content as sample_config.py
5. In config.py set your google gemini api key. 
(create google account -> go to google AI studio -> generate api Key -> copy api key -> paste in config.py)
5. In one terminal run flask api for backend ```python api.py```
6. In a new terminal run streamlit ui ```streamlit run app.py```

## How to Run:
1. Once UI starts, enter company's name in the Option provided.
2. Article will be Analyzed and displayed in UI, Sentiment distribution will be plotted, Hindi Audio summary can be played.
3. Download the json(analysis) and audio(Hindi).

# Assumptions
- Company name provided is appropriate.
- All the articles are extracted from www.bbc.com . Assumption is that article related to the company name provided should be present in already vast library of articles in bbc.com

# Possible Improvements
- Add a checker to verify the company name provided.
- Add more websites to extract links from. This will help reduce bias that a particular new provider might have.
- Hindi speech generated sounds artificial and mechanical. With newer, better models speech can be made to sound more human-like.
- Currently maximum 18 new articles can be extracted, as articles from only first two pages of website are extracted. 
