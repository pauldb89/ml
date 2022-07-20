alias mlbuild="docker build -t ml ."
alias mldoc="docker run --gpus all --shm-size 64G -i -v /home/data1/:/data1 -v /mnt/efs-home/pauldb/datasets:/datasets -v /home/pauldb/code/ml:/ml -v /home/pauldb/.cache:/root/.cache -w /ml -t ml"
