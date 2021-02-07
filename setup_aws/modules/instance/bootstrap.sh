#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Add nvidia-docker repo
source /etc/os-release
distribution="$ID$VERSION_ID"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

# Install dependencies
apt-get update
apt-get install -y \
  awscli \
  docker.io \
  docker-compose \
  jq \
  nvidia-docker2 \
  s3fs

### Ensure data volume is formatted and mounted
mkdir -pm 777 /data
device=$(lsblk -o NAME,FSTYPE -dpsn | awk '$2 == "" {print $1}')
if [[ ! -z "$device" ]]; then
  mkfs.ext4 $device
  echo "$device  /data ext4  defaults,noatime  0 0" >>/etc/fstab
  mount $device /data
fi

### Change docker data location and add runtimes
systemctl stop docker.service

mkdir /data/docker_data
cat <<EOF >/etc/docker/daemon.json
{
  "data-root": "/data/docker_data",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

systemctl start docker.service
usermod -a -G docker ubuntu

# install Nvidia drivers if running on GPU
if lshw | grep -qi nvidia; then
  apt-get install linux-headers-$(uname -r)
  distribution=$(echo $distribution | sed -e 's/\.//g')
  wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.pin
  mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/7fa2af80.pub
  echo "deb http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
  apt-get update
  apt-get -y install cuda-drivers
  export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
  echo 'export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}' >>/home/ubuntu/.profile
  /usr/bin/nvidia-persistenced --verbose
  # test that nvidia docker is working
  docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
fi

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

### extract tfrecords
for percent in 1 3 10 25; do
  tar xf "/mnt/irrigation_data/BigEarthNet_tfrecords/tfrecords_${percent}_percent.tar" \
    -C /data \
    --exclude="tfrecords_${percent}_percent/train.tfrecord" \
    --exclude="tfrecords_${percent}_percent/test.tfrecord" \
    --exclude="tfrecords_${percent}_percent/val.tfrecord" \
    --owner=ubuntu \
    --group=ubuntu \
    --no-same-permissions
done

exit 0
