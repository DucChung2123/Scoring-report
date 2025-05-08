# ESG Text Extraction with New API Integration

This implementation updates the ESG text extraction tool to use the new classification API from `datn/src/api/main.py` instead of the original semantic search approach.

## Key Changes

1. **Local Text Processing**: PDF text extraction now happens directly in the Streamlit app, instead of sending the PDF to the backend API
2. **API Integration**: Uses both classification and scoring endpoints from the new API:
   - `/classify_sub_factor_batch` - For ESG factor and sub-factor classification
   - `/score_batch` - For confidence scoring of extracted text
3. **Sub-factor Mapping**: Includes logic to map between API's sub-factor names and the config's sub-factor names
4. **Result Format**: Maintains compatibility with the existing UI display functions

## How to Use

1. Make sure the ESG Classification API is running:
   ```bash
   cd datn
   python src/api/main.py
   ```
   This will start the API on port 2003.

2. Run the new Streamlit app:
   ```bash
   cd Scoring-report
   streamlit run UI/demo_streamlit_new_api.py
   ```

3. Configure the API URL in the sidebar (default: `http://localhost:2003`)

4. Upload a PDF file and adjust parameters as needed

5. Click "Analyze ESG Data" to process the document

## Differences from Original Implementation

- **Original**: Used semantic search with pre-defined ESG factor explanations to match text
- **New**: Uses a trained ML model to classify text into ESG factors and sub-factors

- **Original**: Sent PDF to backend for processing
- **New**: Processes PDF locally and only sends text snippets to the API

- **Original**: API endpoint was `/api/v1/esg/esg-extract`
- **New**: Uses two API endpoints: `/classify_sub_factor_batch` and `/score_batch`

## Fallback Behavior

- If API calls fail, the app will display appropriate error messages
- For scoring failures, a default score of 0.75 will be used
- For sub-factor mapping issues, text will be assigned to the first available sub-factor for its category
