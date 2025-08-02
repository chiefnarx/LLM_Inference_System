FROM python:3.10-slim

WORKDIR /app

# Copying app files
COPY . /app

# Copying offline packages
COPY wheels /wheels

# Installing packages offline
RUN pip install --no-index --find-links=/wheels -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
