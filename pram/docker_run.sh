IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $IP
docker run --rm -it -v $(pwd):/app -e DISPLAY=$IP:0 --name pram-container pram-app bash
