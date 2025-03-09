FROM python:3.11
WORKDIR /whisperfastapi
COPY ./requirements.txt /whisperfastapi/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /whisperfastapi/requirements.txt
COPY ./app /whisperfastapi/app
COPY ./data /whisperfastapi/data
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
