#include <bits/stdc++.h>

using namespace std;

int isValid(vector<int> & board);

int main(int argc, char ** argv)
{
    int total = 0;
    int n = atoi(argv[1]);

    vector<int> board(n);
    for (int i = 0; i < n; i++)
        board[i] = i;

    do
    {
        total += isValid(board);
    } while (std::next_permutation(board.begin(), board.end()));

    cout << total << endl;
}

int isValid(vector<int> & board)
{
    int is_valid = true;
    for (int i = 0; i < board.size() && is_valid; i++)
    {
        // we are checking this i against the entire board
        int my_height = board[i];
        // check all other peices
        for (int j = 0; j < board.size() && is_valid; j++)
        {
            int dist = abs(i - j);
            if (    (my_height - dist == board[j] || my_height + dist == board[j]) 
                    && j != i)
                is_valid = false;
        }
    }
    return is_valid;
}
