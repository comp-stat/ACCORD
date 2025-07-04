# Dockerfile
FROM python:3.9-slim

# 빌드 도구와 C++ 확장 컴파일을 위한 패키지 설치
RUN apt-get update && apt-get install -y \
    git g++ cmake libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ACCORD v1.1.4 설치 (C++ 확장 포함)
RUN pip install --no-cache-dir git+https://github.com/comp-stat/ACCORD.git@v1.1.4

# accord 명령 실행
ENTRYPOINT ["accord"]
