#!/bin/bash

#SBATCH --job-name=cnn-model-img-gen-gems497
#SBATCH --output=cnn-model-img-gen-gems497-output.txt
#SBATCH --error=cnn-model-img-gen-gems497.err
#SBATCH --time=01:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=class
#SBATCH --partition=class

cd /fs/classhomes/spring2024/gems497/ge497000/keras-ct/final-model/
source /fs/class-projects/spring2024/gems497/ge497g00/team-doc-env/bin/activate
module add cuda/11.8.0 cudnn/v8.8.0
srun bash -c "python3 3dcnn-model-image.py" &
wait
