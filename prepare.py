import subprocess

# run prepare/prepare_polyu.py
subprocess.run(["python", "prepare/prepare_polyu.py"])
# run prepare/prepare_restormer.py
subprocess.run(["python", "prepare/prepare_restormer.py"])
# run prepare/prepare_sidd
subprocess.run(["python", "prepare/prepare_sidd.py"])