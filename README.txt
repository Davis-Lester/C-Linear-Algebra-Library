Linear Algebra Library – Advanced Linear Algebra in C++

This linear algebra library is a lightweight C++ library for 2D matrix operations, including matrix creation, concatenation, inversion, RREF, determinant calculation, and sparse conversion.
Designed for solving voltage equations, this library makes linear algebra intuitive and fast for both dense and sparse matrices.

Features
1. Identity and Zero Matrices
2. Matrix Concatenation (LEFT, RIGHT, TOP, BOTTOM)
3. Truncation (remove rows or columns)
4. Reduced Row Echelon Form (RREF) – solve systems of linear equations
5. Determinant Calculation
6. Transpose & Inverse
7. Sparse Matrix Conversion
8. Operator Overloads (+, -, *)
9. Robust Error Handling

Dense vs Sparse Matrices

Dense Matrix (all elements stored):
Matrix A (4x4):
1 0 0 0
0 5 0 0
0 0 0 2
0 0 0 0

Sparse Matrix (only non-zero elements stored as [row, col, value]):
Sparse(A):
[0,0,1]
[1,1,5]
[2,3,2]

Sparse matrices save memory when most elements are zero.

Preprocessor Choice: Dense vs Sparse
This library allows you to compile your program for dense or sparse matrix operations using preprocessor directives.

Example:
#define USE_SPARSE_MATRIX
#include "Matrix.hpp"

USE_SPARSE_MATRIX → enables sparse matrix optimizations
Not defined → uses normal dense matrix operations
This makes it easy to switch modes without modifying your code logic.


Installation
Include Matrix.hpp and its header in your project:

#include "Matrix.hpp"
using namespace std;

No external dependencies required. Works on any standard C++17 compiler.

Example Usage

#include "Matrix.hpp"

int main() {
    auto I = identityMatrix(3);
    auto Z = zeroMatrix(3, 3);

    auto sum = I + Z;
    auto scaled = sum * 5.0;

    printMatrix(scaled);

    vector<vector<double>> A = {{2,1,-1,8}, {-3,-1,2,-11}, {-2,1,2,-3}};
    auto rrefA = RREF(A);
    printMatrix(rrefA);

    auto sparse = convertToSparse(A);

    return 0;
}