start_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
conda_env="/home/xiaobinz/anaconda3/envs/pytorch-latest/bin/python"
program_dir="/home/xiaobinz/Downloads/horovod-xiaobing/examples/pytorch_dist.py"

/home/xiaobinz/anaconda3/envs/pytorch-latest/bin/mpirun \
    -np 2 \
    -H mlt-skx091:2\
    -bind-to socket -report-bindings -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.20.1/24\
    $conda_env $program_dir : \
    -H mlt-skx060:2\
    -bind-to socket -report-bindings -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    $conda_env $program_dir
finish_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`

duration=$(($(($(date +%s -d "$finish_time")-$(date +%s -d "$start_time")))))
echo "this shell script execution duration: $duration"
