// File name : Matrix.h
// Last edited : 11/2/2025
// Author : Davis Lester
// Description : Header file for Linear Algebra-based Calculations on 2D vectors (Matricies)

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <algorithm>
using namespace std;

// -------------------------------------------------------------------------
// Enumeration Definitions
// -------------------------------------------------------------------------

/**
 * @brief Enumeration for specifying the side of matrix concatenation.
 */
enum class concatonateSide { LEFT, RIGHT, TOP, BOTTOM };

/**
 * @brief Enumeration for specifying whether to truncate rows or columns.
 */
enum class truncateIndex { ROW, COLUMN };

// -------------------------------------------------------------------------
// Function Prototypes (Declarations)
// -------------------------------------------------------------------------

/**
 * @brief Creates an identity matrix of a given size.
 * @param size The number of rows and columns for the square identity matrix.
 * @return vector<vector<double>> The generated identity matrix.
 */
vector<vector<double>> identityMatrix(int size);

/**
 * @brief Creates a matrix full of 0s from a given size
 * * @param rows The number of rows for the matrix.
 * * @param columns The number of columns for the matrix.
 * @return std::vector<std::vector<int>> The generated identity matrix.
 */
vector<vector<double>> zeroMatrix(int rows, int columns);

/**
 * @brief Concatenates two matrices horizontally (LEFT/RIGHT) or vertically (TOP/BOTTOM).
 * @param matrix1 The primary matrix.
 * @param matrix2 The secondary matrix to be appended.
 * @param side The direction of concatenation (LEFT, RIGHT, TOP, or BOTTOM).
 * @return vector<vector<double>> The new combined matrix.
 * @throw runtime_error if matrices have incompatible dimensions.
 */
vector<vector<double>> concatonate(vector<vector<double>> matrix1, vector<vector<double>> matrix2, concatonateSide side);

/**
 * @brief Truncates a matrix by removing rows or columns from the end.
 * @param matrix The matrix to truncate.
 * @param index The number of rows or columns to remove.
 * @param indexType The dimension to truncate (ROW or COLUMN).
 * @return vector<vector<double>> The truncated matrix.
 * @throw runtime_error if index is out of bounds or negative.
 */
vector<vector<double>> truncate(const vector<vector<double>>& matrix, int index, truncateIndex indexType);

/**
 * @brief Solves a system of linear equations represented as a matrix using Gaussian Elimination (RREF).
 * @param matrix The matrix which needs to be solved.
 * @return vector<vector<double>> The solved matrix in Reduced Row Echelon Form.
 * @throw runtime_error if matrix has 0 rows.
 */
vector<vector<double>> RREF(vector<vector<double>> matrix);

/**
 * @brief Removes a row and column from a larger square matrix to form a submatrix (minor).
 * @param matrix The matrix to have a row and column removed.
 * @param rowToRemove The row index to be removed.
 * @param colToRemove The column index to be removed.
 * @return vector<vector<double>> The submatrix.
 */
vector<vector<double>> getSubmatrix(const vector<vector<double>>& matrix, int rowToRemove, int colToRemove);

/**
 * @brief Computes the determinant of a square matrix using Cofactor Expansion.
 * @param matrix The matrix to find the determinant of.
 * @return double The determinant of the matrix.
 * @throw runtime_error if matrix is not square.
 */
double calculateDeterminant(vector<vector<double>> matrix);

/**
 * @brief Computes the transpose of a matrix.
 * @param matrix The matrix to find the transpose of.
 * @return vector<vector<double>> The transpose matrix.
 * @throw runtime_error if matrix is empty.
 */
vector<vector<double>> transpose(vector<vector<double>> matrix);

/**
 * @brief Computes the inverse matrix of a square matrix using the augmented matrix (A|I -> I|A^-1) method.
 * @param matrix The matrix to find the inverse of.
 * @return vector<vector<double>> The inverse matrix.
 * @throw runtime_error if matrix is not square or empty.
 */
vector<vector<double>> inverse(vector<vector<double>> matrix);

/**
 * @brief Displays a matrix to the terminal.
 * @param matrix The matrix to print.
 * @throw runtime_error if matrix is empty.
 */
void printMatrix(vector<vector<double>> matrix);

/**
 * @brief Converts a standard matrix into a simplified sparse representation (COO format).
 * @param matrix The standard matrix.
 * @return vector<vector<double>> The sparse matrix (3 rows: R, C, Value).
 * @throw runtime_error if matrix is empty.
 */
vector<vector<double>> convertToSparse(vector<vector<double>> matrix);

// -------------------------------------------------------------------------
// Operator Overloads
// -------------------------------------------------------------------------

/**
 * @brief Overloads the * operator for matrix multiplication.
 * @param matrix1 The left-hand matrix.
 * @param matrix2 The right-hand matrix.
 * @return vector<vector<double>> The result of the multiplication.
 * @throw runtime_error if dimensions are incompatible.
 */
vector<vector<double>> operator*(const vector<vector<double>> matrix1, const vector<vector<double>> matrix2);

/**
 * @brief Overloads the * operator for scalar multiplication (matrix * double).
 * @param matrix The matrix.
 * @param num The scalar value.
 * @return vector<vector<double>> The result of the multiplication.
 */
vector<vector<double>> operator*(const vector<vector<double>> matrix, const double num);

/**
 * @brief Overloads the + operator for matrix addition.
 * @param matrix1 The left-hand matrix.
 * @param matrix2 The right-hand matrix.
 * @return vector<vector<double>> The result of the addition.
 * @throw runtime_error if dimensions are incompatible.
 */
vector<vector<double>> operator+(const vector<vector<double>> matrix1, const vector<vector<double>> matrix2);

/**
 * @brief Overloads the - operator for matrix subtraction.
 * @param matrix1 The left-hand matrix.
 * @param matrix2 The right-hand matrix.
 * @return vector<vector<double>> The result of the subtraction.
 * @throw runtime_error if dimensions are incompatible.
 */
vector<vector<double>> operator-(const vector<vector<double>> matrix1, const vector<vector<double>> matrix2);

#endif // MATRIX_H