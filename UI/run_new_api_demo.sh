#!/bin/bash

# Colors for console output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================================${NC}"
echo -e "${GREEN}ESG Classification System - New API Implementation${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Check if the required directories exist
if [ ! -d "../../datn" ]; then
    echo -e "${RED}Error: datn directory not found.${NC}"
    echo -e "${YELLOW}Please make sure you're running this script from the Scoring-report/UI directory.${NC}"
    exit 1
fi

# Define the API and Streamlit log paths
API_LOG="./api_server.log"
STREAMLIT_LOG="./streamlit_server.log"

echo -e "${BLUE}Starting ESG Classification API...${NC}"

# Start the API server in the background
cd ../../datn || exit
python src/api/main.py > "$API_LOG" 2>&1 &
API_PID=$!

# Give the API server time to start
echo -e "${YELLOW}Waiting for API server to initialize (5 seconds)...${NC}"
sleep 5

# Check if API server is running
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}API server started successfully! (PID: $API_PID)${NC}"
else
    echo -e "${RED}Failed to start API server. Check $API_LOG for details${NC}"
    exit 1
fi

# Return to the Scoring-report directory to run Streamlit
cd ../Scoring-report || exit

echo -e "${BLUE}Starting Streamlit application...${NC}"

# Run Streamlit in the foreground
streamlit run UI/demo_streamlit_new_api.py

# When Streamlit is terminated, also terminate the API server
echo -e "${YELLOW}Shutting down API server (PID: $API_PID)...${NC}"
kill $API_PID

echo -e "${GREEN}Cleanup complete. Goodbye!${NC}"
