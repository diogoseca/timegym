# Development image with the recorded scientific stack; distributions use flit.
FROM python:3.15-slim
WORKDIR /app
COPY requirements-study.txt .
RUN python -m pip install --no-cache-dir -r requirements-study.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps -e .
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
CMD ["python", "-m", "pytest", "-q"]
