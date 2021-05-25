#docker build -t timegym-image .
#docker run --rm -it -p 8888:8888 -v "$PWD":/home/jovyan/work timegym-image
docker run -it -p 8888:8888 -v "$PWD":/home/jovyan/work timegym-image