SHELL       = /bin/bash
BUILDX      = docker buildx build --push --platform linux/amd64,linux/arm64
DOCKER_REPO = imander


.PHONY: build
build: tf24

.PHONY: tf24
tf24:
	$(BUILDX) -t $(DOCKER_REPO)/irgapp -f tf24.Dockerfile .
