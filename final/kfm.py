#!/usr/bin/env python3

import subprocess
import random
import datetime


subprocess.run(["make", "parallel"], universal_newlines=True)
subprocess.run(["make", "serial"], universal_newlines=True)

print("f_e,p")

for i in range(3, 17):
	
	# RUN PARALLEL #
	start = datetime.datetime.now()		
	p_output = subprocess.run(["mpirun", "-np", str(i), "./nqueens", "11", "0", "1"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	end = datetime.datetime.now()
	p_runtime = end - start


	# RUN SERIAL #
	start = datetime.datetime.now()		
	s_output = subprocess.run(["./serial", "11"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	end = datetime.datetime.now()
	s_runtime = end - start

	# kfm = (1/speedup - 1/p) / (1 - 1/p)
	# speedup = 1/(1-p_runtime + (p_runtime/s_runtime))
	speedup = (s_runtime - p_runtime) / p_runtime
	print(str( abs((1/speedup - 1/i) / (1 - 1/i))) + "," + str(i))
