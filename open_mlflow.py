# opens the mlflow page in the browser (right-click > Run in pycharm)
# stop it with the red square (Stop) button in pycharm when done

import subprocess
import sys
import threading
import webbrowser

import settings

# the server needs a few seconds to start, so open the browser after 5 s
threading.Timer(5, webbrowser.open, ["http://127.0.0.1:5000"]).start()

# same as typing: mlflow ui --backend-store-uri sqlite:///mlflow.db
# sys.executable = the python of the patientfinding environment
subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", settings.MLFLOW_URI])
