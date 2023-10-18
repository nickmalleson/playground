# To create the pram-docker container (also create Dockerfile)

docker build -t pram-app .

# To run the image

docker run -d -p 8000:8000 --name pram-container pram-app

# To stop the image

docker stop pram-container

# Install XQuartz and, in settings->security, tell it to 'Allow connections from network clients'

# Then do the following (in XQuartz terminal) to get the current IP and allow to accept connections

IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $IP


# Run the container. Note: 
#  - with a volume mount (so that we can edit py files and changes are reflected in the vm)
#  - `bash` at the end means you drop out into a shell rather than having the VM exit
#  - connect the X display to the mac for displaying figures

#docker run -it -p 8000:8000 -v $(pwd):/app --name pram-container pram-app bash
docker run -it -v $(pwd):/app -e DISPLAY=$IP:0 --name pram-container pram-app bash


