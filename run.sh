#!/bin/bash

log_dir="./log/"
seed=123
lr=0.0001
weight_decay=0.00005
epochs=1500
output=2048
cluster=50
num_neighbors=100

if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

dataset=$1

case $dataset in
  "PubMed")
    lr=0.00005
    weight_decay=0.0005
    output=4096
    epochs=1500
    cluster=50
    num_neighbors=100
    ;;
  "CS")
    lr=0.0001
    weight_decay=0.00005
    output=2048
    epochs=1500
    cluster=50
    num_neighbors=100
    ;;
  "Photo")
    lr=0.00001
    weight_decay=0.00001
    output=4096
    epochs=600
    cluster=30
    num_neighbors=100
    ;;
  "Computers")
    lr=0.00005
    weight_decay=0.00001
    output=4096
    epochs=200
    cluster=30
    num_neighbors=100
    ;;
  "Physics")
    lr=0.00001
    weight_decay=0.00005
    output=2048
    epochs=600
    cluster=15
    num_neighbors=100
    ;;
  "Wiki-CS")
    lr=0.00001
    weight_decay=0.00005
    output=512
    epochs=200
    cluster=15
    num_neighbors=10
    ;;
  *)
    echo "Unknown dataset: $dataset"
    exit 1
    ;;
esac

python ./train.py --dataset $dataset --log_dir $log_dir --seed $seed --output $output --lr $lr --weight_decay $weight_decay --epochs $epochs --cluster $cluster --num_neighbors $num_neighbors
