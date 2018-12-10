#!/usr/bin/env python3

import subprocess
import random
import datetime


subprocess.run(["make"], stdout=subprocess.PIPE, universal_newlines=True)


for i in range(1, 17):
	start = datetime.datetime.now()		
	output = subprocess.run(["mpirun", "-np", "15", "./parallel", str(i), "0"], universal_newlines=True).stdout
	end = datetime.datetime.now()
	runtime = end - start
	print(str(i) + "X" + str(i) + "\n" + "Processors: 15" + " took: " + str(runtime.total_seconds()) + " microseconds\n" + output)
