FROM python:3.10
EXPOSE 8501
WORKDIR /app
# COPY requirements.txt ./requirements.txt
# RUN pip3 install -r requirements.txt

COPY pyproject.toml poetry.lock ./

RUN pip install poetry==1.5.0 && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    poetry shell

COPY . .

ENTRYPOINT ["streamlit"]
CMD ["run", "myapp.py"]
