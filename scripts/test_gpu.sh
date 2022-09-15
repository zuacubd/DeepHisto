#!/bin/sh

#SBATCH --job-name=VN16_Te
#SBATCH --output=logs/vgg16_wsi_classification_test.out
#SBATCH --error=logs/vgg16_wsi_classification_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

echo "starting .."
#srun singularity exec /projets/sig/mullah/singularity/sif/ubuntu18_osirim.sif python3 "../programs/main.py" -extract TRUE -level 0 -size 224 -overlap FALSE
srun singularity exec /logiciels/containerCollections/CUDA9/keras-tf.sif python3 "../programs/main.py" -test True -network vgg16
echo "done"
