#!/bin/bash -l
# specify a partition
#SBATCH -p batch
# specify number of nodes 
#SBATCH -N 4
# specify number of cores
#SBATCH -n 16
# specify the job name
#SBATCH -J kmeans
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/tnagan/kmeans/slurm.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/tnagan/kmeans/slurm.err

file=credit_card_data.csv

echo ------------------------------------------------------
echo Job is running on node $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR
make clean
make
echo Running Serial K-means
echo Changing iterations
for i in {50..500..50}
do
    echo "Number of Iterations = $i"
    ./kmeans $file 2 3 $i
    echo
done

echo Changing dimensions 
for i in {2..17..1}
do
    echo "Number of Dimensions = $i"
    ./kmeans $file $i 3 100
    echo
done

echo Changing K 
for i in {2..5..1}
do
    echo "Number of K = $i"
    ./kmeans $file 2 $i 100
    echo
done

echo Running cuda K-means
echo Changing iterations
for i in {50..500..50}
do
    echo "Number of Iterations = $i"
    ./kmeans_cuda $file 2 3 $i
    echo
done

echo Changing dimensions 
for i in {2..17..1}
do
    echo "Number of Dimensions = $i"
    ./kmeans_cuda $file $i 3 100
    echo
done

echo Changing K 
for i in {2..5..1}
do
    echo "Number of K = $i"
    ./kmeans_cuda $file 2 $i 100
    echo
done

echo Running MPI K-means
echo Changing iterations
for i in {50..500..50}
do
    echo "Number of Iterations = $i"
    srun --mpi=pmi2 --ntasks-per-node=4 --ntasks=16 --nodes=4 kmeans_mpi $file 2 3 $i
    echo
done

echo Changing dimensions 
for i in {2..17..1}
do
    echo "Number of Dimensions = $i"
    srun --mpi=pmi2 --ntasks-per-node=4 --ntasks=16 --nodes=4 kmeans_mpi $file $i 3 100
    echo
done

echo Changing K 
for i in {2..5..1}
do
    echo "Number of K = $i"
    srun --mpi=pmi2 --ntasks-per-node=4 --ntasks=16 --nodes=4 kmeans_mpi $file 2 $i 100
    echo
done
echo ------------------------------------------------------
echo Kmeans Complete!
echo ------------------------------------------------------
