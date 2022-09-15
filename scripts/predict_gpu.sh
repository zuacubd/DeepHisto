#!/bin/sh

#SBATCH --job-name=vgg19_Predict
#SBATCH --output=logs/vgg19_classification_prediction.out
#SBATCH --error=logs/vgg19_classification_prediction.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

echo "starting .."
#srun singularity exec /projets/sig/mullah/singularity/sif/ubuntu18_osirim.sif python3 "../programs/main.py" -extract TRUE -level 0 -size 224 -overlap FALSE
srun singularity exec /logiciels/containerCollections/CUDA9/keras-tf.sif python3 "mc_programs/main.py" -predict True -network vgg19
echo "done"
