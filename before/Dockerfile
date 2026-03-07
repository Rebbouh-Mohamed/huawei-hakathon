# Start with Python already installed
FROM python:3.9-slim

# Install supervisor (the remote control for running multiple things)
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Create a workspace inside the box
WORKDIR /app

# Copy your requirements list
COPY requirements.txt .

# Install all the Python packages you need
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL your files into the box
COPY . .

# Create a folder for logs
RUN mkdir -p /var/log/supervisor

# When the box starts, run supervisor
CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"]