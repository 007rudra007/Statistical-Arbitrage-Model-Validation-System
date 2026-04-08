# ── Aladdin Spark Pipeline – Docker Image ────────────────────────────────────
# Extends the base Python 3.12 slim image with PySpark + Delta Lake + S3A
# Build: docker build -f services/spark/Dockerfile -t aladdin-spark .
# Run:   docker run --rm -e MINIO_ENDPOINT=http://host.docker.internal:9000 aladdin-spark

FROM python:3.12-slim AS base

# ── Java (required by Spark) ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV DELTA_VERSION=3.1.0

# ── Download Spark ────────────────────────────────────────────────────────────
RUN curl -fsSL \
    "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xz -C /opt \
    && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3

# ── Download Delta Lake + S3A JARs ───────────────────────────────────────────
# These are placed directly in Spark's jars dir so they auto-load.
ARG JAR_DIR=/opt/spark/jars

RUN curl -fsSL \
    "https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/${DELTA_VERSION}/delta-spark_2.12-${DELTA_VERSION}.jar" \
    -o "${JAR_DIR}/delta-spark.jar" && \
    curl -fsSL \
    "https://repo1.maven.org/maven2/io/delta/delta-storage/${DELTA_VERSION}/delta-storage-${DELTA_VERSION}.jar" \
    -o "${JAR_DIR}/delta-storage.jar" && \
    curl -fsSL \
    "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar" \
    -o "${JAR_DIR}/hadoop-aws.jar" && \
    curl -fsSL \
    "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar" \
    -o "${JAR_DIR}/aws-java-sdk-bundle.jar"

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
        pyspark==${SPARK_VERSION} \
        delta-spark==${DELTA_VERSION} \
        pyarrow>=16.0.0 \
        boto3>=1.34.0 \
        yfinance>=0.2.18 \
        pandas>=2.0 \
        numpy>=1.24

# ── Application code ──────────────────────────────────────────────────────────
COPY data/        ./data/
COPY services/spark/ ./services/spark/

ENV PYTHONPATH=/app

# Default entrypoint: run the Spark pipeline
CMD ["python", "services/spark/spark_pipeline.py"]
