import sys
import os

# Ajout du dossier du projet au path
path = '/home/GuiM78/P7Scoring'
if path not in sys.path:
    sys.path.append(path)

# Définition du bon répertoire de travail
os.chdir(path)

# Import de l'application
from app import app as application