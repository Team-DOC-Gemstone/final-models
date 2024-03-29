#!/bin/bash

#SBATCH --job-name=cnn-demo-gems497
#SBATCH --output=cnn-demo-gems497-output.txt
#SBATCH --error=cnn-demo-gems497.err
#SBATCH --time=06:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=class
#SBATCH --partition=class

current_name=`date +"%Y-%m-%d-%H-%M-%S"`.txt
cd /fs/classhomes/spring2024/gems497/ge497000/
source /fs/class-projects/spring2024/gems497/ge497g00/team-doc-env/bin/activate
module add cuda/11.8.0 cudnn/v8.8.0
srun bash -c "python3 ~/cnn-demo.py" &
wait
