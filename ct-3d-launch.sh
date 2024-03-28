#!/bin/bash

#SBATCH --job-name=keras-lc-ct-gems497
#SBATCH --output=output-keras-lc-ct-gems497.txt
#SBATCH --error=err-keras-lc-ct-gems497.err
#SBATCH --time=10:00:00
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=class
#SBATCH --partition=class

cd /fs/classhomes/spring2024/gems497/ge497000/keras-ct/final-model
source /fs/classhomes/spring2024/gems497/ge497000/team-doc/bin/activate
module add cuda/11.8.0 cudnn/v8.8.0
srun bash -c "python3 doc_3d_convnet.py" &
wait
