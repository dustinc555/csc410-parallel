
mull1: mull_1.cu
	nvcc -std=c++11 mull_1.cu -o parallel

mull2: mull_2.cu
	nvcc -std=c++11 mull_2.cu -o parallel

serial: mull_serial.cpp
	g++ -std=c++11 -DTIMED mull_serial.cpp -o serial

mull1Debug: mull_1.cu
	nvcc -std=c++11 -DDEBUG mull_1.cu -o parallel

mull2Debug: mull_2.cu
	nvcc -std=c++11 -DDEBUG mull_2.cu -o parallel 


mull1Timed: mull_1.cu
	nvcc -std=c++11 -DTIMED mull_1.cu -o parallel

mull2Timed: mull_2.cu
	nvcc -std=c++11 -DTIMED mull_2.cu -o parallel


serialTest: mull_serial.cpp
	g++ -std=c++11 -DTEST mull_serial.cpp -o serial

mull1Test: mull_1.cu
	nvcc -std=c++11 -DTEST mull_1.cu -o parallel

test: mull_1.cu mull_serial.cpp 
	make serialTest
	make mull1Test

timed: mull_1.cu mull_serial.cpp
	make serial
	make mull1Timed
clean:
	@rm -f serial parallel
