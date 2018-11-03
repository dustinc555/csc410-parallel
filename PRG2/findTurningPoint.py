#!/usr/bin/env python3

import subprocess
import re

for i in range(1, 10000, 2):
	output_p = subprocess.run(["./parallel", "50", str(i), "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", "50", "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	time_p = int(output_p)
	time_s = int(output_s)
	if time_p - time_s > 0:	
		print("turing point at row size = " + str(i))
	if time_p == time_s:
		print("The same! at row size = " + str(i))
