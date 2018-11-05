#!/usr/bin/env python3

import subprocess
import re

subprocess.run(["make", "clean"], stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run(["make", "all"], stdout=subprocess.PIPE, universal_newlines=True)

for i in range(1, 10000, 2):
	output_p = subprocess.run(["./parallel", str(i), "t"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", str(i), "t", "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	print(output_p + "," + output_s)
