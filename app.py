import streamlit as st
import requests
from PIL import Image
import io
import json

# # Flask backend URL
FLASK_URL = "http://127.0.0.1:5000/analyze"

# Streamlit UI
st.set_page_config(layout="centered")
st.title("Company News Sentiment Analysis Tool")

# # Input for company name
company_name = st.text_input("Enter the company name:")

# Initialize session state for report
if "report" not in st.session_state:
    st.session_state.report = None

if company_name:
    if st.session_state.report is None or st.session_state.report.get("company_name") != company_name:
        st.write(f"Fetching news articles for {company_name}...")

        # Send request to Flask backend
        response = requests.post(FLASK_URL, json={"company_name": company_name})
        print(response)

        if response.status_code == 200:
            # Store the report in session state
            st.session_state.report = response.json()
            st.session_state.report["company_name"] = company_name  # Add company name to report
        else:
            st.error("Error fetching data from the server. Please try again.")
            st.session_state.report = None

    if st.session_state.report:
        report = st.session_state.report

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

        # Audio and Download Button in Columns
        st.subheader("Hindi Audio Output")
        col1, col2 = st.columns(2)
        with col1:
            st.audio(report["Audio"])
        with col2:
            # Construct the full URL for the audio file
            audio_url = f"http://127.0.0.1:5000/{report['Audio']}"
            audio_bytes = requests.get(audio_url).content
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="audio_output.mp3",
                mime="audio/mpeg"
            )

        # Display results
        st.subheader("Sentiment Analysis Results")
        st.json(report)  # Use st.json for better JSON display

        # Download JSON Report Button
        json_report = json.dumps(report, indent=4)
        st.download_button(
            label="Download JSON Report",
            data=json_report,
            file_name="sentiment_analysis_report.json",
            mime="application/json"
        )

else:
    st.write("Please enter a company name to get started.")