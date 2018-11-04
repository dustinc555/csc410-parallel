#!/usr/bin/env python3

import subprocess
import re


subprocess.run(["make", "clean"], stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run(["make", "timed"], stdout=subprocess.PIPE, universal_newlines=True)

for i in range(1, 10000, 2):
	output_p = subprocess.run(["./parallel", str(i), "1", "0"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", str(i), "1"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	time_p = int(output_p)
	time_s = int(output_s)
	if time_p < time_s:	
		print("turing point at row size = " + str(i))
	if time_p == time_s:
		print("The same! at row size = " + str(i))
