
FROM python:3.9-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV PORT 8080

# Run app.py when the container launches
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT