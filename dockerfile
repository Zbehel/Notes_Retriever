# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/
COPY app.py /app
COPY pyproject.toml /app
COPY uv.lock /app

# Install uv
RUN pip install uv

# Create a virtual environment with uv
RUN uv venv .venv

# Activate the virtual environment. This is important for subsequent commands.
ENV PATH="/app/.venv/bin:$PATH"

# Install project dependencies from pyproject.toml and uv.lock
#RUN uv pip install --no-cache-dir -r /app/uv.lock
RUN uv sync --locked
# Make port 8501 available to the world outside this container
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]
