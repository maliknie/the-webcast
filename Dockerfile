# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install uv and dependencies
RUN pip install --no-cache-dir uv \
    && pip install --no-cache-dir playsound \
    && uv pip install --no-cache-dir -r requirements.txt --system

# Copy the rest of the application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Disable Streamlit usage stats
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the app
CMD ["streamlit", "run", "./app/streamlit_ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
