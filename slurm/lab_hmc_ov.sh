#!/bin/bash
#SBATCH --job-name=lab          # Nombre del trabajo
#SBATCH --chdir=/home/ptflores1/storage/UAI
#SBATCH --output=logs/lab_hmc_ov.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=logs/lab_hmc_ov.log          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --time=8-00:00:00            # Timpo limite d-hrs:min:sec

#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ptflores1@uc.cl    # El mail del usuario



#SBATCH --dependency=singleton
#SBATCH --partition=ialab-high

date;hostname;pwd

source /home/ptflores1/storage/UAI/venv/bin/activate
python3 main.py lab hmc   -ov  

date