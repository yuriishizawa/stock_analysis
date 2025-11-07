FROM python:3.10
EXPOSE 8501
WORKDIR /app

# Create a non-root user with dedicated group
RUN addgroup --system appuser && \
    adduser --system --home /app --no-create-home --ingroup appuser appuser

# COPY requirements.txt ./requirements.txt
# RUN pip3 install -r requirements.txt

COPY pyproject.toml poetry.lock ./

RUN pip install poetry==1.5.0 && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    poetry shell

COPY . .

# Change ownership of application files to non-root user
RUN chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["streamlit"]
CMD ["run", "myapp.py"]
