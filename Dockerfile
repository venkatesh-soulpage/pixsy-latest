FROM python:3.6.12

RUN apt-get update

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 9099

CMD ["uvicorn", "server.app:app",  "--host", "0.0.0.0", "--port", "9099"]
