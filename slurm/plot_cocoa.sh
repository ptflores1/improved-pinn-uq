#!/bin/bash
#SBATCH --job-name=plot_eq_cocoa          # Nombre del trabajo
#SBATCH --chdir=/home/ptflores1/storage/UAI
#SBATCH --output=logs/plot_eq_cocoa.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=logs/plot_eq_cocoa.log          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --time=1-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ptflores1@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high
#SBATCH --mem=64G
#SBATCH --exclude=hydra

date;hostname;pwd

source /home/ptflores1/storage/UAI/venv/bin/activate
python3 -m plotters.cocoa

date