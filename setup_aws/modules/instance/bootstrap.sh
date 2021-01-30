#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Install dependencies
apt-get update
apt-get install -y \
  awscli \
  docker.io \
  docker-compose \
  jq \
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
usermod -a -G docker ubuntu

### Mount S3
mkdir -pm 777 /mnt/irrigation_data
s3fs mids-capstone-irrigation-detection /mnt/irrigation_data -o iam_role=auto -o allow_other

### Get Github API token
github_token=$(aws secretsmanager get-secret-value \
  --region us-west-2 \
  --secret-id 'arn:aws:secretsmanager:us-west-2:672750028551:secret:github-5LKXx9' |
  jq -r '.SecretString' |
  jq -r '.gh_api_token')

### Set auth file and clone repos
sudo -iu ubuntu /bin/bash - <<EOF
echo -e "machine github.com\nlogin mids-capstone-irrigation\npassword $github_token" >~/.netrc
chmod 600 ~/.netrc

mkdir repos
cd repos
git clone https://github.com/MIDS-Irrigation-Capstone/Spring2021.git
git clone https://github.com/MIDS-Irrigation-Capstone/Fall2020.git
EOF

### Pull irrigation docker image
docker pull imander/irgapp

exit 0
