#!/bin/bash
#SBATCH --job-name=lab_bundle_inverse          # Nombre del trabajo
#SBATCH --chdir=/home/ptflores1/storage/UAI
#SBATCH --output=logs/lab_inverse_lab_nlm_bundle.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=logs/lab_inverse_lab_nlm_bundle.log          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=16            # Numero de cores por tarea
#SBATCH --time=8-00:00:00            # Timpo limite d-hrs:min:sec

#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ptflores1@uc.cl    # El mail del usuario



#SBATCH --dependency=singleton
#SBATCH --partition=ialab-high

date;hostname;pwd

source /home/ptflores1/storage/UAI/venv/bin/activate
python3 main.py lab nlm -b   -ids=lab  -i=2

date