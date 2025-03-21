import sys
import os

# Ajouter le dossier de l'API au sys.path
path = '/home/GuiM78/P7Scoring/api'  
if path not in sys.path:
    sys.path.append(path)

# Définir le bon répertoire de travail
os.chdir(path)

# Importer l'application Flask
from app import app as application
