#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Install dependencies
apt-get update
apt-get install -y \
  docker.io \
  docker-compose \
  s3fs

### Ensure data volumne is fomatted and mounted
mkdir -pm 777 /data
lsblk /dev/nvme1n1 | grep -q ext4 ||
  (
    mkfs.ext4 /dev/nvme1n1
    echo '/dev/nvme1n1  /data ext4  defaults,noatime  0 0' >>/etc/fstab
  )

mount /dev/nvme1n1 /data

### Change docker data location
systemctl stop docker.service
echo '{"data-root": "/data"}' >/etc/docker/daemon.json
systemctl start docker.service

### Mount S3
mkdir -pm 777 /mnt/irrigation_data
s3fs mids-capstone-irrigation-detection /mnt/irrigation_data -o iam_role=auto -o allow_other

exit 0
