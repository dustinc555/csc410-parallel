#!/usr/bin/env python3

import subprocess
import re

for i in range(490000, 622000, 2000):
	output_p = subprocess.run(["./parallel", "700", str(i), "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", "700", "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	# f_e = (1/speedup - 1/p) / (1 - 1/p)
	# parallel program should output total_threads time_nanoseconds
	# serial program should output time_nanoseconds
	time_p = int(output_p)
	time_s = int(output_s)
	speedup = (time_s - time_p) / time_s
	pInv = 1.0/i
	print("f_e = " + str(((1/speedup) - pInv) / (1 - (pInv))) + ", P = " + str(i))
	
