import streamlit as st
import pandas as pd
import json
import requests

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

# Sidebar
with st.sidebar:
    st.header("Upload & Settings")
    
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    # Parameters
    st.subheader("Parameters")
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum similarity score to include a result (0-1)"
    )
    
    top_k = st.number_input(
        "Top K Results",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Number of top results to return for each factor"
    )
    
    # API endpoint setting
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8888/api/v1/esg/esg-extract"
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

# Function to analyze the PDF file through API
def analyze_pdf(file, threshold, top_k, api_url):
    # Create form data
    files = {'upload_file': file}
    params = {
        'threshold': threshold,
        'top_k': top_k
    }
    try:
        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None

# Function to display results
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

# Function to display results for a specific ESG category
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

# Main content area
if uploaded_file is not None and analyze_button:
    with st.spinner("Analyzing document for ESG information..."):
        results = analyze_pdf(
            uploaded_file,
            threshold,
            top_k,
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
    2. Adjust analysis parameters if needed
    3. Click "Analyze ESG Data" to start the analysis
    4. View the detailed results across E, S, and G categories
    
    The tool will identify relevant ESG information in your document and score it based on similarity to established ESG criteria.
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
