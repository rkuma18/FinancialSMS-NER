"""Streamlit client for FastAPI NER inference service."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import requests
import streamlit as st

if TYPE_CHECKING:
    # Type Hinting ke liye (runtime mein yeh ignore ho jaate hain)
    from typing import Dict, List
    Entity = Dict[str, str | float | int]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---

# ****** YAHAN PAR CHANGE KIYA GAYA HAI ******
# Render Deployment ke liye, API URL ko hardcode kiya gaya hai.
# 'financial-sms-ner-app.onrender.com' ko aapke actual live URL se badalna na bhulein!
RENDER_BASE_URL = "https://financial-sms-ner-app.onrender.com" 
API_URL = f"{RENDER_BASE_URL}/predict"

# Purana Docker Compose Line:
# API_URL = os.getenv("API_URL", "http://backend:8000/predict")

DEFAULT_SMS = (
    "Rs. 5000 credited to A/c XXXXX1234 on 15/09/2025 at 10:30 PM by HDFC Bank. OTP 123456."
)
HIGHLIGHT_STYLE = (
    "background-color:#e0f7fa;padding:2px 4px;border-radius:4px;"
    "border:1px solid #00bcd4;color:#006064;font-weight:600;"
)


# --- Helper Functions (Defined before main() to resolve NameError) ---

def fetch_entities(text: str) -> List[Entity]:
    """Call the FastAPI NER prediction endpoint."""
    # Ensure API_URL is correct before making request
    logger.info(f"Sending request to API_URL: {API_URL}")
    response = requests.post(API_URL, json={"text": text}, timeout=30)
    # Agar status code 4xx ya 5xx hai toh exception raise hoga
    response.raise_for_status() 
    return response.json().get("entities", [])


def highlight_text(text: str, entities: List[Entity]) -> str:
    """Highlight entities in text with HTML/CSS."""
    # Reverse sort zaroori hai taki replacement ke dauraan character indices shift na ho
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        start, end = ent["start"], ent["end"]
        
        # Entity tag ke saath styling
        highlight = (
            f"<span style='{HIGHLIGHT_STYLE}'>"
            f"{ent['word']} <small style='color:#006064;'>({ent['entity_group']})</small></span>"
        )
        text = text[:start] + highlight + text[end:]
    return text


def display_results(text: str, entities: List[Entity]) -> None:
    """Display extracted entities in table and highlighted text."""
    st.divider()
    
    st.markdown("### Extracted Entities Table")
    df_data = [
        {
            "Entity": e["entity_group"],
            "Text": e["word"],
            "Score": f"{e['score']:.4f}",
            "Start": e["start"],
            "End": e["end"],
        }
        for e in entities
    ]
    st.dataframe(df_data, use_container_width=True, hide_index=True)
    
    st.markdown("### Highlighted Text")
    # Streamlit ko batana padta hai ki yeh HTML code hai
    st.markdown(highlight_text(text, entities), unsafe_allow_html=True)


# --- Main Application Entry Point ---

def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="SMS NER Demo", layout="centered")
    st.title("Transactional SMS Named Entity Recognition 🤖")
    
    user_text = st.text_area("Enter Transactional SMS Text", value=DEFAULT_SMS, height=140)
    
    # Button press hone tak aage ka code execute nahi hoga
    if not st.button("Extract Entities", type="primary"):
        return
    
    text = user_text.strip()
    if not text:
        st.warning("Please enter some text to analyze.")
        return
    
    # API Call aur Error Handling
    try:
        with st.spinner("Calling NER inference API..."):
            entities = fetch_entities(text)
            
    except requests.HTTPError as exc:
        # API ne 4xx ya 5xx error diya (jaise 400 Bad Request, 503 Service Unavailable)
        st.error(f"API Error: {exc.response.status_code} - {exc.response.text}")
        logger.error("API HTTP Error", exc_info=True)
        return
        
    except requests.RequestException:
        # Connection error (jaise server down hai ya network issue)
        st.error(f"Failed to reach API at {API_URL}. Is the server running?")
        logger.error("API Connection Error", exc_info=True)
        return
    
    if not entities:
        st.info("No entities detected in the provided text.")
        return
    
    # Results Display
    display_results(text, entities)


if __name__ == "__main__":
    main()