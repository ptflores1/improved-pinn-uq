#!/bin/bash
#SBATCH --job-name=metrics_latex          # Nombre del trabajo
#SBATCH --chdir=/home/ptflores1/storage/UAI
#SBATCH --output=logs/metrics_latex.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=logs/metrics_latex.log          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=2            # Numero de cores por tarea
#SBATCH --time=5-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ptflores1@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high
#SBATCH --mem=20G

date;hostname;pwd

source /home/ptflores1/storage/UAI/venv/bin/activate
python3 -m metrics.latex

date