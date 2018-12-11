#!/usr/bin/env python3

import csv


spamReader = csv.reader(open('plot.csv'))

for row in spamReader:
	p = 12
	pTime = float(row[1])
	sTime = float(row[2])
	speedup = sTime / pTime
	print( str((speedup - p) / (1 - p)) )
