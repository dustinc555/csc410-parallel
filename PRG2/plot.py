#!/usr/bin/env python3

import subprocess
import re

for i in range(1, 10000, 2):
	output_p = subprocess.run(["./parallel", "50", "r", str(i)], stdout=subprocess.PIPE, universal_newlines=True).stdout
	output_s = subprocess.run(["./serial", "50", "r"], stdout=subprocess.PIPE, universal_newlines=True).stdout
	print(output_p + "," + output_s)
