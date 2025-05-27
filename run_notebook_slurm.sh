#!/bin/bash
#SBATCH --job-name=notebook          # Nombre del trabajo
##SBATCH --workdir=/home/ptflores1/storage/UAI
#SBATCH --output=logs/notebook_%j.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=logs/notebook_%j.log          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --time=8-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --partition=ialab-high

pwd; hostname; date

source /home/ptflores1/storage/UAI/venv/bin/activate
jupyter notebook --no-browser --ip="*" --port=1337


echo "Finished with job $SLURM_JOBID"