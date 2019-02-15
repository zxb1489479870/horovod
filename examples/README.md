## Horovod
1. Install OpenMPI: https://www.open-mpi.org/faq/?category=building#easy-build
2. Install torch(the given exampe has runtime error for lates pytorch, pytorch=0.4.1 can be work)

```bash
$conda install pytorch=0.4.1 -c pytorch
```
3. install horovod
```bash
pip install horovod or pip install --no-cache-dir horovod
```
or 
```bash
pip install git+https://github.com/uber/horovod@master
```
4. Run the bash script: ``` ./run_horovod.sh```

## torch.distributed
1. Install the optional dependencies
```bash
# In case using conda virtual environment, point this to corresponding virtual env directory
# e.g. export CMAKE_PREFIX_PATH=/home/user/anaconda/env/virtual_env_name/
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```
2. Install OpenMPI using source code: https://www.open-mpi.org/faq/?category=building#easy-build

Note: there some errors if install OpenMPI according to https://pytorch.org/tutorials/intermediate/dist_tuto.html

3. Install Pytorch using source code: ``` pyhton setup.py install```

4. run the bash script:``` ./run_torch.sh```
## How to using mpirun: http://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/
