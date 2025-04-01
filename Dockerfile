FROM langchain/langgraph-api:3.11



# -- Installing local requirements --
ADD requirements.txt /deps/__outer_workout_AssIstant/src/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -r /deps/__outer_workout_AssIstant/src/requirements.txt
# -- End of local requirements install --

# -- Adding non-package dependency workout_AssIstant --
ADD . /deps/__outer_workout_AssIstant/src
RUN set -ex && \
    for line in '[project]' \
                'name = "workout_AssIstant"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_workout_AssIstant/pyproject.toml; \
    done
# -- End of non-package dependency workout_AssIstant --

# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"memory-agent": "/deps/__outer_workout_AssIstant/src/workout_AssIstant.py:graph"}'

WORKDIR /deps/__outer_workout_AssIstant/src