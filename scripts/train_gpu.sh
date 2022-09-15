#!/bin/sh

#SBATCH --job-name=resnet101
#SBATCH --output=logs/resnet101_wsi_classification_tr_te_rp.out
#SBATCH --error=logs/resnet101_wsi_classification_tr_te_rp.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding

echo "starting .."
#srun singularity exec /projets/sig/mullah/singularity/sif/ubuntu18_osirim.sif python3 "../programs/main.py" -extract TRUE -level 0 -size 224 -overlap FALSE
srun singularity exec /logiciels/containerCollections/CUDA9/keras-tf.sif python3 "mc_programs/main.py" -train True -network resnet101
echo "done"
