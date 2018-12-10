
#include <cuda.h>
#include <cuda_runtime.h>


 __global__ void isValid(int * board, int SIZE, bool * is_valid)
{
        extern __shared__ int s[];

        // this can actually be simplified but i wanted to use cuda
        if (threadIdx.x < SIZE) return;
        int i = threadIdx.x;
        int my_height = board[i];


        s[i] = true;

        for (int j = 0; j < SIZE && is_valid; j++)
        {
                int dist = abs(i - j);
                if (    (my_height - dist == board[j] || my_height + dist == board[j])
                        && j != i)
                        s[j] = false;
        }

        if (i == 0)
                for (int j = 0; j < SIZE; j++)
                        *is_valid *= s[j]; // if there is a single zero the solution is false
}

extern "C" void launch_valid(int * board, int SIZE, bool * is_valid)
{
	// allocate and transfer data to device
        int * d_board;
	bool * d_is_valid;
        int size = SIZE * sizeof(int);
        cudaMalloc( (void**) &d_board, size);
        cudaMemcpy( board, d_board, size, cudaMemcpyHostToDevice  );

        // allocate is_valid as well
        cudaMalloc( (void**) &d_is_valid, sizeof(bool) );
        cudaMemcpy( is_valid, d_is_valid, sizeof(bool), cudaMemcpyHostToDevice );
	
	// execute kernel
        int threads = SIZE + (SIZE % 32);
        
	//cout << "threads: " << threads << endl;
        isValid<<<1, threads>>>(d_board, SIZE, d_is_valid);

	// get result back from device
	cudaMemcpy( is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost );
}

