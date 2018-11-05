#!/usr/bin/env python3

import subprocess
import random


subprocess.run(["make", "clean"], stdout=subprocess.PIPE, universal_newlines=True)
subprocess.run(["make", "all"], stdout=subprocess.PIPE, universal_newlines=True)


for i in range(1, 10):
	file = open("input.txt", "w")
	matrix = ""
	vector = ""
	for j in range(i**2):
		matrix += str(round(random.uniform(1, 10), 2)) + " "
	matrix += "\n"

	for j in range(i):
		vector += str(round(random.uniform(1, 10), 2)) + " "
	vector += "\n"
	file.write(matrix)
	file.write(vector)
	file.close()
		
	output_p = subprocess.run(["./parallel", str(i), "1"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", str(i), "1"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	print("Parallel result:\n" + output_p + "\n\nSerial Result:\n" + output_s + "\n")

