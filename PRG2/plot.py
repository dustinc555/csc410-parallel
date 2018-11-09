#!/usr/bin/env python3

import subprocess
import re

subprocess.run(["make", "clean"], stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run(["make", "all"], stdout=subprocess.PIPE, universal_newlines=True)

for i in range(1000, 2000, 25):
	output_p = subprocess.run(["./parallel", str(i), "t"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", str(i), "t", "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	better = "not better"
	if int(output_p) - int(output_s) < 0:
 		better = "better"
	if int(output_p) - int(output_s) == 0:
		better = "same"
	print(output_p + "," + output_s + "," + better + "," + str(i))
