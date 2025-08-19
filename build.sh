# #!/bin/bash
# # Create a "tag" or name for the image
# docker_tag=aicregistry:5000/${USER}-image-mmpose:latest

# # build the image calling on your Dockerfile
# docker build . -f Dockerfile \
#  --network=host \
#  --tag ${docker_tag} \
#  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER}
# docker push ${docker_tag}
# # docker_tag=aicregistry:5000/${USER}:image-mmpose

docker_tag=aicregistry:5000/${USER}-robustmip

echo $docker_tag

docker build . -f Dockerfile --tag ${docker_tag} --network=host --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER}

docker push $docker_tag