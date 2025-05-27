#!/bin/bash
#SBATCH --job-name=plot_data_cocoa__          # Nombre del trabajo
#SBATCH --chdir=/home/ptflores1/storage/UAI
#SBATCH --output=logs/plot_data_cocoa___test.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=logs/plot_data_cocoa___test.log          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=8            # Numero de cores por tarea
#SBATCH --time=5-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ptflores1@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high
#SBATCH --mem=64G
#SBATCH --dependency=singleton

date;hostname;pwd

source /home/ptflores1/storage/UAI/venv/bin/activate
python3 -m plotters.main cocoa    --domain_type=test

date