#!/usr/bin/env python3

import subprocess
import random
import datetime


subprocess.run(["make", "parallel"], universal_newlines=True)
subprocess.run(["make", "serial"], universal_newlines=True)

print("n,p,s")

for i in range(1, 17):
	
	# RUN PARALLEL #
	start = datetime.datetime.now()		
	p_output = subprocess.run(["mpirun", "-np", "12", "./nqueens", str(i), "0", "1"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	end = datetime.datetime.now()
	parallel_runtime = end - start


	# RUN SERIAL #
	start = datetime.datetime.now()		
	s_output = subprocess.run(["./serial", str(i)], stdout=subprocess.PIPE, universal_newlines=True).stdout
	end = datetime.datetime.now()
	serial_runtime = end - start

	print(str(i) + "," + str(parallel_runtime.total_seconds()) + "," + str(serial_runtime.total_seconds()))
