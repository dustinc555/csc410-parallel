#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>

#include <omp.h>
#include <mpi.h>

using namespace std;

void permutor(int p, int n, int threads);
void collector(int p);
void slave(int p, int n, int id, int o, int threads);
int factorial(int n);

int permutorID;
int collectorID;

int main(int argc, char **argv)
{
	int id; // rank of executing process
	int p;	// number of processes
	int permutorID = 0;
	int collectorID = p - 1;
	int n = atoi(argv[1]);  
	int o = atoi(argv[2]); // 0 to not print 1 to print
	int thread_count = atoi(argv[3]);

	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	permutorID = 0;
	collectorID = p - 1;

       
	if (id == 0)			permutor(p, n, thread_count);
	else if (id == p - 1)	collector(p);
	else					slave(p, n, id, o, thread_count);	


	MPI_Finalize();
	return 0;
}

void permutor(int p, int n, int threads)
{

	/*
	int len;
	char p_name[100];
        MPI_Get_processor_name(p_name, &len);
        cout << p_name << " assigned as permutor" << endl;
	*/

	// make intial (sorted) array, send it
	vector<int> board(n);
	for (int i = 0; i < n; i++)
		board[i] = i;
	
	int i;
	int j = 1; // used to send messages
	int thread_count = threads;

	
	#pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1) 
	for (i = 0; i < board.size(); i++)
	{
		int slaveID = (i % (p - 2)) + 1;

		// the diagonal rotations are not required, only the first one for case 1.
		vector<int> vprime = board;

		std::rotate(vprime.begin(), vprime.begin() + i, vprime.begin() + i + 1);

		MPI_Send(	&vprime[0],
				vprime.size(),
				MPI_INT,
				slaveID,
				0,
				MPI_COMM_WORLD	);
	}

	cout << "ALL ROTATIONS SENT" << endl;

	// send quit message to slaves
	board[0] = -1;
	for (int i = 1; i < p - 1; i++)
	{
		MPI_Send(       &board[0],
                                board.size(),
                                MPI_INT,
                                i,
                                0,
                                MPI_COMM_WORLD	);
		
	}

}


void collector(int p)
{
	/*
	int len;
	char p_name[100];
        MPI_Get_processor_name(p_name, &len);
        cout << p_name << " assigned as collector" << endl;
	*/

	vector<bool> slaves(p);
	slaves[p - 1] = false;
	slaves[0] = false;
	
	/* set a flag bit for every slave
	e.x. p = 4, 
		0 = permutor
		3 = permutor
		1, 2 = slaves
		we get: 0110
	*/
	int flags = (1 << (p - 1)) - 2; 

	//std::bitset<32> y(flags);
	//cout << "Flags: " << y << endl;

	int solutions = 0;
	int is_solution;


	// While we are not done and 	
	while (flags) // as long as there are active slave servers
	{
		MPI_Recv(	&is_solution,
				1,
				MPI_INT,
				MPI_ANY_SOURCE,
				0,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE	);
		//cout << "Collector received solution: " << is_solution << " solutions: " << solutions << endl;
		if (is_solution < 0) {
			//std::bitset<32> y((int) (pow(2, (abs(is_solution)))));
        		//cout << "Received: " << y << endl;	
			flags &= ~( 1 << (abs(is_solution))); // this slave has stopped working, flip its switch
		}
		else
			solutions += is_solution;

		//cout << "Flags: " << flags << endl;
	}

	cout << "Total Solutions: " << solutions << endl;
}

void slave(int p, int n, int id, int o, int threads)
{
	/*
	int len;
	char p_name[100];
        MPI_Get_processor_name(p_name, &len);
	cout << p_name << " assigned as slave" << endl;
	*/

	int collectorID = p - 1;
	int permutorID = 0;
	bool working = true;
	int total_solutions = 0;
	// receive a board
	// for each element in board
	//	if (not out of range)
	// 		if neighbor == elem +- dist 
	vector<int> board(n);

	int permutes = factorial(n-1); // each slave handles one lexigraphical set of permutes
	//cout << "permutes: " << permutes << " n: " << n << endl;


	while (working)
	{
		int hasDone = 0;

		MPI_Recv(	&board[0],
				n,
				MPI_INT,
				permutorID,
				0,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE	);

		if (board[0] == -1)	break; // we are done

		/*
		cout << p_name << " received job: ";
		for (int i = 0; i < n; i++)
			cout << board[i]<< " ";
		cout << endl;
		*/      

		// is_valid starts true tell proven false
		int thread_count = threads;

		for (int k = 0; k < permutes; k++)
		{
			/*
			cout << p_name << " working on: ";
			for (int i = 0; i < n; i++)
				cout << board[i]<< " ";
			cout << endl;
			*/
			bool is_valid = true;

			#pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1) reduction(=:is_valid)
			for (int i = 0; i < n && is_valid; i++)
			{
				// we are checking this i against the entire board
				int my_height = board[i];
				/*
				cout << "my i: " << i << endl;
				cout << "my_height: " << my_height << endl;
				*/
				// check all other peices
				for (int j = 0; j < n && is_valid; j++)
				{
					int dist = abs(i - j);
					//cout << "i: " << i << " j: " << j << " dist: " << dist << endl;
					/* to be diagonal means concurrent distance 
					it has a slope of exactly 1 or -1 with another element
					example: board[3] = 2, and board[2] = 1
					distance from pos 3 and pos 2 is 1,
								(board[3] + or - dist) == 1 
					this holds true to the end */
					//cout << "minus case: " << my_height - dist == board[i] << endl;
					//cout << "plus case: " << my_height + dist == board[i] << endl;
					
					if (    (my_height - dist == board[j] || my_height + dist == board[j]) 
							&& j != i)
						is_valid = false;
				
				}
			}
				
			if (is_valid)
			{
				//cout << "sending: " << is_valid << " to " << collectorID << endl;
				if (o)
				{
					cout << "Solution found: <";
					for (int i = 0; i < board.size(); i++)
						cout << board[i] << " ";
					cout << ">\n";
				}

				// if this is a valid solution sends 1 else 0
				total_solutions++; 
			}

			std::next_permutation(board.begin(), board.end()); 
		}
	}

	MPI_Send(       &total_solutions,
        		1,
                        MPI_INT,
                        collectorID,
                        0,
                        MPI_COMM_WORLD  );

	int shutdown = -id; // tell collector we are done
	MPI_Send(	&shutdown,
					1,
					MPI_INT,
					collectorID,
					0,
					MPI_COMM_WORLD	);

}


int factorial(int n)
{
	int f = n;
	for (int i = 2; i < n; i++)
		f *= i;
	return f;
}

