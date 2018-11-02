#!/usr/bin/env python3

import subprocess

for i in range(1, 1001):
	output = subprocess.run(["./parallel", str(i), "1"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	for c in (output.split()):
		if (c != str(i)):
			print("wrong: " + str(i))
			break
	else:
		print("correct: " + str(i))
	
