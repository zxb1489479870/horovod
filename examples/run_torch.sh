conda_env="/home/xiaobinz/anaconda3/envs/pytorch-latest/bin/python"
program_dir="/home/xiaobinz/Downloads/horovod-xiaobing/examples/pytorch_dist.py"

/home/xiaobinz/anaconda3/envs/pytorch-latest/bin/mpirun \
    -np 2 \
    -H mlt-skx121:2\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.20.1/24\
    $conda_env $program_dir : \
    -H mlt-skx054:2\
    -bind-to socket -report-bindings -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    $conda_env $program_dir

