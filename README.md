docker pull tensorflow/serving

docker run -d -p 8501:8501 \
--name=nsfw
--mount type=bind,source=`pwd`/data/models,target=/models/nsfw \
-e MODEL_NAME=nsfw -t tensorflow/serving