SHELL       = /bin/bash
BUILDX      = docker buildx build --push --platform linux/amd64,linux/arm64
DOCKER_REPO = imander


.PHONY: build
build: tf24 tf23

.PHONY: tf24
tf24:
	$(BUILDX) -t $(DOCKER_REPO)/irgapp:2.4 -f tf24.Dockerfile .

.PHONY: tf23
tf23:
	$(BUILDX) -t $(DOCKER_REPO)/irgapp:2.3 -f tf23.Dockerfile .
