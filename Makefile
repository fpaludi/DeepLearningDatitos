IMAGE_NAME="pytorch_dev"
CONTAINER_NAME="ctr_pytorch_dev"
SRC_VOLUME="$(PWD)/src/:/workspace/src"
DATA_VOLUME="$(PWD)/data:/workspace/data"
JUP_PORT=4016
BOARD_PORT=6006

build:
	docker build -t $(IMAGE_NAME) -f ./docker/Dockerfile .

run_dev:
	docker run -it \
	--rm \
	--privileged \
	--name $(CONTAINER_NAME) \
	-p $(JUP_PORT):$(JUP_PORT) -p $(BOARD_PORT):$(BOARD_PORT) \
	--memory=4g \
	-v $(SRC_VOLUME) \
	-v $(DATA_VOLUME) \
	$(IMAGE_NAME) \
	bash


