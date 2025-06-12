import streamlit as st
import pandas as pd
import json
import requests
import sys
import os
from pathlib import Path

# Add parent directory to sys.path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api.services.text.extraction import PDFTextExtractor

# Set page configuration
st.set_page_config(
    page_title="ESG Text Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS to improve the UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .category-header {
        font-size: 1.6rem;
        padding: 0.5rem 0;
        margin-top: 1rem;
    }
    .factor-header {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .text-card {
        background-color: #F9FAFB;
        border-left: 4px solid #4B5563;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    .text-card-e {
        border-left: 4px solid #059669;
    }
    .text-card-s {
        border-left: 4px solid #2563EB;
    }
    .text-card-g {
        border-left: 4px solid #7C3AED;
    }
    .score-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        color: white;
        margin-bottom: 0.5rem;
    }
    .score-badge-e {
        background-color: #059669;
    }
    .score-badge-s {
        background-color: #2563EB;
    }
    .score-badge-g {
        background-color: #7C3AED;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">ESG Text Viewer</h1>', unsafe_allow_html=True)

# Load factor mapping from config
def load_esg_factors():
    try:
        # Use the finetune config directly since settings might not be available
        from src.finetune.extract_utils.config import SUB_FACTORS_E, SUB_FACTORS_S, SUB_FACTORS_G
        return {
            'e': {factor: f"{factor} related content" for factor in SUB_FACTORS_E},
            's': {factor: f"{factor} related content" for factor in SUB_FACTORS_S}, 
            'g': {factor: f"{factor} related content" for factor in SUB_FACTORS_G}
        }
    except ImportError:
        # Fallback if import fails
        return {
            'e': {
                'Emission': 'Emissions related content',
                'Resource Use': 'Resource usage content',
                'Product Innovation': 'Product innovation content'
            },
            's': {
                'Community': 'Community related content',
                'Diversity': 'Diversity related content',
                'Employment': 'Employment related content',
                'HS': 'Health and safety content', 
                'HR': 'Human rights content',
                'PR': 'Product responsibility content',
                'Training': 'Training related content'
            },
            'g': {
                'BFunction': 'Board function content',
                'BStructure': 'Board structure content',
                'Compensation': 'Compensation related content',
                'Shareholder': 'Shareholder related content',
                'Vision': 'Vision related content'
            }
        }

# Map between API sub-factors and config sub-factors
def map_sub_factor_to_config(factor, sub_factor, esg_factors):
    # Convert factor to lowercase for matching
    factor_lower = factor.lower()
    
    # If the factor is not E, S, or G, return None
    if factor_lower not in ['e', 's', 'g']:
        return None, None
    
    # Get all sub-factors for this factor from config
    config_sub_factors = list(esg_factors[factor_lower].keys())
    
    # Find the best match for the API sub-factor in the config sub-factors
    best_match = None
    
    # Special case for "Others" - if no other match is found
    if sub_factor.lower() == "others":
        # Just use the first sub-factor as a fallback
        return factor_lower, config_sub_factors[0]
    
    # Try direct match first
    for config_sub in config_sub_factors:
        if sub_factor.lower() == config_sub.lower():
            return factor_lower, config_sub
    
    # Try partial match if direct match fails
    for config_sub in config_sub_factors:
        if sub_factor.lower() in config_sub.lower() or config_sub.lower() in sub_factor.lower():
            best_match = config_sub
            break
    
    # If we found a match, return it
    if best_match:
        return factor_lower, best_match
    
    # If no match found, use the first sub-factor as a fallback
    return factor_lower, config_sub_factors[0]

# Function to call classify_sub_factor_batch API
def classify_paragraphs(paragraphs, api_url):
    """
    Send paragraphs to the classify_sub_factor_batch API endpoint.
    
    Args:
        paragraphs (list): List of text paragraphs to classify
        api_url (str): Base URL for the API
        
    Returns:
        list: List of classification results
    """
    # Updated endpoint path to match the new API structure
    endpoint = f"{api_url}/classify_sub_factor_batch"
    
    # Process paragraphs in batches to avoid overwhelming the API
    batch_size = 32
    all_results = []
    
    # Show a progress bar for classification
    batch_progress = st.progress(0)
    batch_status = st.empty()
    
    for i in range(0, len(paragraphs), batch_size):
        # Get the current batch
        batch = paragraphs[i:i+batch_size]
        
        # Update status
        batch_progress.progress(i / len(paragraphs))
        batch_status.text(f"Classifying paragraphs {i+1}-{min(i+batch_size, len(paragraphs))} of {len(paragraphs)}...")
        
        # Create payload for this batch - updated to match new API schema
        payload = [{"text": para} for para in batch]
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            batch_results = response.json()
            
            # Validate the response format
            if not isinstance(batch_results, list):
                st.error(f"Unexpected response format from API: {batch_results}")
                continue
                
            all_results.extend(batch_results)
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling classification API: {str(e)}")
            # Continue with other batches instead of failing completely
    
    # Clean up progress indicators
    batch_progress.empty()
    batch_status.empty()
    
    return all_results

# Function to score paragraphs for specific ESG factors
def score_paragraphs(paragraphs, factors, api_url):
    """
    Score paragraphs for specific ESG factors.
    
    Args:
        paragraphs (list): List of text paragraphs to score
        factors (list): List of factors to score (E, S, G)
        api_url (str): Base URL for the API
        
    Returns:
        list: List of scoring results
    """
    # Updated endpoint path to match the new API structure
    endpoint = f"{api_url}/score_batch"
    
    # Process paragraphs in batches to avoid overwhelming the API
    batch_size = 20
    all_results = []
    
    # Create all requests first (paragraph + factor combinations)
    all_requests = []
    for para in paragraphs:
        for factor in factors:
            all_requests.append({"text": para, "factor": factor})
    
    # Show a progress bar for scoring
    score_progress = st.progress(0)
    score_status = st.empty()
    
    # Process in batches
    for i in range(0, len(all_requests), batch_size):
        # Get the current batch
        batch = all_requests[i:i+batch_size]
        
        # Update status
        progress_percent = i / len(all_requests)
        score_progress.progress(progress_percent)
        score_status.text(f"Scoring batch {i // batch_size + 1} of {(len(all_requests) // batch_size) + 1}...")
        
        try:
            response = requests.post(endpoint, json=batch)
            response.raise_for_status()
            batch_results = response.json()
            
            # Validate response format
            if not isinstance(batch_results, list):
                st.error(f"Unexpected response format from API: {batch_results}")
                # Use default scores instead
                batch_results = [{"score": 0.75} for _ in batch]
                
            all_results.extend(batch_results)
        except requests.exceptions.RequestException as e:
            st.warning(f"Error scoring batch: {str(e)}. Using default scores.")
            # Use default scores if API call fails
            all_results.extend([{"score": 0.75} for _ in batch])
    
    # Clean up progress indicators
    score_progress.empty()
    score_status.empty()
    
    return all_results

# Function to analyze the PDF file through the new API
def analyze_pdf(file, api_url):
    """
    Process PDF file and analyze using the new ESG classification API.
    
    Args:
        file: The uploaded PDF file
        api_url: Base URL for the API
        
    Returns:
        dict: Structured results for display
    """
    try:
        # Save uploaded file to a temporary location
        os.makedirs("uploads", exist_ok=True)
        file_name = file.name
        file_path = os.path.join("uploads", file_name)
        with open(file_path, "wb") as buffer:
            buffer.write(file.read())  # Use read() for Streamlit's UploadedFile
        
        # Extract and clean text from PDF
        json_texts = PDFTextExtractor.extract_text_from_pdf(file_path)
        paragraphs = PDFTextExtractor.clean_vietnamese_text(json_texts)
        
        if not paragraphs:
            st.error("Could not extract text from the PDF file.")
            return None
            
        # Progress indicator for long processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get ESG factors from config
        esg_factors = load_esg_factors()
        
        # Step 1: Classify all paragraphs
        status_text.text("Classifying paragraphs...")
        classifications = classify_paragraphs(paragraphs, api_url)
        progress_bar.progress(0.5)  # Use 0.5 for 50%
        
        if not classifications:
            st.error("Failed to classify paragraphs. Check API connection.")
            return None
        
        # Organize results by ESG category and sub-factor
        esg_results = {'e': {}, 's': {}, 'g': {}}
        
        # Initialize sub-factor dictionaries
        for factor, subfactors in esg_factors.items():
            for subfactor in subfactors:
                esg_results[factor][subfactor] = []
        
        # Collect all paragraphs that need to be scored
        score_requests = []
        para_mappings = []  # Store mapping info for each request
        
        status_text.text("Preparing text for scoring...")
        
        # First, organize all paragraphs that need scoring
        for i, (paragraph, classification) in enumerate(zip(paragraphs, classifications)):
            factor = classification.get('factor', '').lower()
            sub_factor = classification.get('sub_factor', '')
            
            # Skip if not E, S, G
            if factor not in ['e', 's', 'g']:
                continue
                
            # Map sub-factor to config keys
            mapped_factor, mapped_sub = map_sub_factor_to_config(factor, sub_factor, esg_factors)
            
            if mapped_factor and mapped_sub:
                # Add to scoring requests
                score_requests.append({
                    "text": paragraph,
                    "factor": mapped_factor.upper()
                })
                
                # Store mapping info
                para_mappings.append({
                    "paragraph": paragraph,
                    "factor": mapped_factor,
                    "sub_factor": mapped_sub
                })
        
        # Now batch process all scoring requests
        status_text.text("Scoring paragraphs...")
        scores = []
        
        # Process in batches of 20
        batch_size = 20
        for i in range(0, len(score_requests), batch_size):
            # Update progress bar
            progress_percent = 0.5 + ((i / len(score_requests)) * 0.5)
            progress_bar.progress(min(progress_percent, 1.0))
            
            # Get current batch
            batch = score_requests[i:i+batch_size]
            status_text.text(f"Scoring batch {i//batch_size + 1} of {len(score_requests)//batch_size + 1}...")
            
            try:
                # Updated endpoint path to match new API structure
                score_response = requests.post(f"{api_url}/score_batch", json=batch)
                score_response.raise_for_status()
                batch_scores = score_response.json()
                scores.extend(batch_scores)
            except Exception as e:
                # Use default scores if API call fails
                st.warning(f"Batch scoring failed, using defaults: {str(e)}")
                scores.extend([{"score": 0.75} for _ in batch])
        
        # Process the results
        for i, mapping in enumerate(para_mappings):
            if i < len(scores):
                score = scores[i]["score"]
                esg_results[mapping["factor"]][mapping["sub_factor"]].append([
                    mapping["paragraph"], score
                ])
        
        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")
        
        # Remove progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Prepare final response
        response = {
            "file_name": file_name,
            "total_paragraphs": len(paragraphs),
            "esg_results": esg_results
        }
        
        return response
        
    except Exception as e:
        st.error(f"Error analyzing PDF: {str(e)}")
        return None

# Function to display results (unchanged from original)
def display_results(results):
    # Extract data
    file_name = results['file_name']
    total_paragraphs = results['total_paragraphs']
    esg_results = results['esg_results']
    
    # Document info
    st.markdown(f"**Document**: {file_name}")
    
    # ESG tabs
    esg_tabs = st.tabs(["Environmental 🌿", "Social 👥", "Governance 📊"])
    
    # Tab content for Environmental
    with esg_tabs[0]:
        if 'e' in esg_results:
            display_category_results(esg_results['e'], "e", "Environmental")
        else:
            st.info("No Environmental data available.")
    
    # Tab content for Social
    with esg_tabs[1]:
        if 's' in esg_results:
            display_category_results(esg_results['s'], "s", "Social")
        else:
            st.info("No Social data available.")
    
    # Tab content for Governance
    with esg_tabs[2]:
        if 'g' in esg_results:
            display_category_results(esg_results['g'], "g", "Governance")
        else:
            st.info("No Governance data available.")
    
    # Export options
    st.download_button(
        label="Download Results as JSON",
        data=json.dumps(results, indent=2),
        file_name=f"esg_analysis_{file_name}.json",
        mime="application/json"
    )

# Function to display results for a specific ESG category (unchanged from original)
def display_category_results(category_data, category_code, category_name):
    if not category_data:
        st.info(f"No {category_name} factors found.")
        return
    
    st.markdown(f'<h2 class="category-header">{category_name} Factors</h2>', unsafe_allow_html=True)
    
    # Iterate through each factor
    for factor_name, results in category_data.items():
        if not results:
            continue
            
        # Create an expander for each factor
        with st.expander(f"{factor_name} ({len(results)} results)", expanded=True):
            # Display factor name header
            st.markdown(f'<div class="factor-header">{factor_name}</div>', unsafe_allow_html=True)
            
            # Display each text item
            for text_item in results:
                text = text_item[0]
                score = text_item[1]
                
                # Create a colored card for each text item
                st.markdown(
                    f"""
                    <div class="text-card text-card-{category_code}">
                        <div class="score-badge score-badge-{category_code}">Score: {score:.2f}</div>
                        <div>{text}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# Sidebar
with st.sidebar:
    st.header("Upload & Settings")
    
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    # No parameters section since the model directly classifies text
    
    # API endpoint setting
    st.subheader("API Settings")
    
    # Auto-detect API URL: use environment variable if in Docker, localhost if running locally
    default_api_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    
    api_url = st.text_input(
        "ESG Classification API URL",
        value=default_api_url,
        help="Auto-detected from environment. Change if needed."
    )
    
    # Add analyze button in sidebar
    analyze_button = st.button("Analyze ESG Data", type="primary")
    
    st.divider()
    
    # About section
    with st.expander("About"):
        st.markdown("""
        This dashboard analyzes PDF documents for Environmental, Social, and Governance (ESG) content.
        
        It uses natural language processing and semantic search to identify relevant ESG information in your documents.
        
        **Categories analyzed:**
        - **E**: Environment (Emissions, Resource Use, Product Innovation)
        - **S**: Social (Community, Diversity, Employment, HR, PR, etc.)
        - **G**: Governance (Board Functions, Structure, Compensation, etc.)
        """)

# Colors for ESG
esg_colors = {
    'e': {'primary': '#059669', 'light': '#D1FAE5'},
    's': {'primary': '#2563EB', 'light': '#DBEAFE'},
    'g': {'primary': '#7C3AED', 'light': '#EDE9FE'},
}

# Main content area
if uploaded_file is not None and analyze_button:
    with st.spinner("Analyzing document for ESG information..."):
        results = analyze_pdf(
            uploaded_file,
            api_url
        )
        
        if results:
            display_results(results)
            
            # Add to session state to maintain results across reruns
            st.session_state.last_results = results
        else:
            st.error("Failed to analyze document. Please check logs and try again.")
elif 'last_results' in st.session_state:
    # Display cached results
    display_results(st.session_state.last_results)
else:
    # Landing page
    st.markdown("""
    ## ESG Scoring Analysis Tool
    
    Welcome to the ESG Scoring Report Dashboard. This tool analyzes PDF documents for Environmental, Social, and Governance (ESG) content.
    
    ### How to use:
    1. Upload a PDF document using the sidebar
    2. Enter the API URL if different from default
    3. Click "Analyze ESG Data" to start the analysis
    4. View the detailed results across E, S, and G categories
    
    The tool uses a machine learning model to identify and classify ESG-related content in your document.
    """)
    
    # Sample images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Environmental 🌿")
        st.markdown("- Emissions and Energy")
        st.markdown("- Resource Use")
        st.markdown("- Climate Innovation")
    
    with col2:
        st.markdown("### Social 👥")
        st.markdown("- Community Relations")
        st.markdown("- Diversity & Inclusion")
        st.markdown("- Employee Wellbeing")
    
    with col3:
        st.markdown("### Governance 📊")
        st.markdown("- Board Structure")
        st.markdown("- Ethics & Transparency")
        st.markdown("- Shareholder Rights")

# Footer
st.markdown('<div class="footer">ESG Analysis Dashboard - Created with Streamlit</div>', unsafe_allow_html=True)
