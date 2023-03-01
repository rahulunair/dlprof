#!/usr/bin/env bash

CORES=$(lscpu | grep Core\(s\) | awk '{print $4}')
SOCKETS=$(lscpu | grep Socket | awk '{print $2}')
TOTAL_CORES=$(expr $CORES \* $SOCKETS)
HALF_CORES=$(expr $CORES \/ 2)
HLAST_CORE=$(expr $HALF_CORES - 1)
LAST_CORE=$(expr $CORES - 1)
NUMA="numactl --physcpubind=0-$LAST_CORE --membind=0"
KMP_BLOCKTIME=0  # 1 , you can tune this as well, the number is the thread idle time before sleep in ms.
INPUT_FILE=$1

export "KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo "## num cores: $CORES"
echo "## num sockets: $SOCKETS"
echo "## using KMP_BLOCKTIME=$KMP_BLOCKTIME"

echo "## single thread test"
echo "## OMP_NUM_THREADS=1"
NUMA="numactl --physcpubind=0 --membind=0"
echo -e "## numa config $NUMA\n"
OMP_NUM_THREADS=1 $NUMA python -m torch.utils.bottleneck $INPUT_FILE

NUMA="numactl --physcpubind=0-$HLAST_CORE --membind=0"
echo "## half socket test"
echo "## OMP_NUM_THREADS=$HALF_CORES"
echo "## numa config=$NUMA\n"
OMP_NUM_THREADS=$HALF_CORES $NUMA python -m torch.utils.bottleneck $INPUT_FILE

NUMA="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo "## single socket test"
echo "## OMP_NUM_THREADS=$CORES"
echo "## numa config=$NUMA\n"
OMP_NUM_THREADS=$CORES $NUMA python -m torch.utils.bottleneck $INPUT_FILE

echo "## all socket test"
echo "## OMP_NUM_THREADS=$TOTAL_CORES"
OMP_NUM_THREADS=$TOTAL_CORES python -m torch.utils.bottleneck $INPUT_FILE
