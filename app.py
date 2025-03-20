import streamlit as st
import requests
from PIL import Image
import io

# Flask backend URL
FLASK_URL = "http://127.0.0.1:5000/analyze"

# Streamlit UI
st.title("Company News Sentiment Analysis Tool")

# Input for company name
company_name = st.text_input("Enter the company name:")

if company_name:
    st.write(f"Fetching news articles for {company_name}...")

    # Send request to Flask backend
    response = requests.post(FLASK_URL, json={"company_name": company_name})
    print(response)

    if response.status_code == 200:
        report = response.json()

        # Final Sentiment
        st.subheader("Overall Sentiment")
        st.write(report["Final Sentiment Analysis"])

        # Fetch the plot from the Flask backend
        plot_url = "http://127.0.0.1:5000/plot"
        plot_response = requests.post(plot_url, json=report)

        # Plot 1
        if plot_response.status_code == 200:
            # Convert the image bytes to a PIL Image
            image = Image.open(io.BytesIO(plot_response.content))
            st.image(image, caption="Sentiment Distribution")

        # Audio
        st.subheader("Hindi Audio Output")
        st.audio(report["Audio"])

        # Display results
        st.subheader("Sentiment Analysis Results")
        st.write(report)
                

    else:
        st.error("Error fetching data from the server. Please try again.")
else:
    st.write("Please enter a company name to get started.")