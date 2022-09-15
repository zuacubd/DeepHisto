#!/bin/bash

#SBATCH --job-name=Patch-extraction
#SBATCH --output=logs/patch-extraction.out
#SBATCH --error=logs/patch-extraction.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=64CPUNodes
#SBATCH --mem-per-cpu=7168M

echo "starting .."
srun singularity exec /projets/sig/mullah/singularity/sif/ubuntu18_osirim.sif python3 "mc_programs/main.py" -extract True -level 0 -size 224 -overlap False -ext jpeg
#srun singularity exec /projets/sig/mullah/singularity/sif/ubuntu18_osirim.sif python3 "../programs/main.py" -combine True -criteria probability
#srun singularity exec /logiciels/containerCollections/CUDA9/keras-tf.sif python3 "../programs/main.py" -combine True -criteria probability
echo "done"
