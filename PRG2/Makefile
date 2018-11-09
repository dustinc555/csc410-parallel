
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

serialDebug: mull_serial.cpp
	g++ -std=c++11 -DDEBUG mull_serial.cpp -o serial

all: mull_1.cu mull_serial.cpp
	make mull1
	make serial

allDebug: mull_1.cu mull_serial.cpp
	make mull1Debug
	make serialDebug
clean:
	@rm -f serial parallel
