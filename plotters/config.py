import os
import matplotlib.pyplot as plt
import matplotlib

#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
os.environ["PATH"] += os.pathsep + '/home/ptflores1/storage/tex-installation/bin/x86_64-linux'
plt.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=20)