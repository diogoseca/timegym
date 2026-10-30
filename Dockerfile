FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir -e '.[dev]'
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
CMD ["python", "-m", "pytest", "-q"]
