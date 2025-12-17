// File name : SparseMatrix.h
// Last edited : 11/2/2025
// Author : Davis Lester
// Description : Header file for Sparse Matrix linear algebra calculations (COO format).

#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "Matrix.hpp" 
using namespace std;

// -------------------------------------------------------------------------
// Structure and Enumeration Definitions
// -------------------------------------------------------------------------

/**
 * @brief The Coordinate List (COO) structure for representing a sparse matrix.
 * * It stores the full dimensions and three parallel vectors for nonzero elements:
 * row_indices (R), col_indices (C), and values.
 */
struct SparseMatrixCOO {
    size_t R; // Full number of rows
    size_t C; // Full number of columns
    vector<double> row_indices; 
    vector<double> col_indices; 
    vector<double> values;      
};

/**
 * @brief Enumeration for specifying the side of sparse matrix concatenation.
 */
enum class sparseConcatonateSide { LEFT, RIGHT, TOP, BOTTOM };

/**
 * @brief Enumeration for specifying whether to truncate rows or columns.
 */
enum class sparseTruncateIndex { ROW, COLUMN };

// -------------------------------------------------------------------------
// Helper Function Prototypes (Declarations)
// -------------------------------------------------------------------------

/**
 * @brief Converts a dense matrix (vector<vector<double>>) to the SparseMatrixCOO structure.
 * @param matrix The dense matrix.
 * @return SparseMatrixCOO The sparse matrix representation.
 */
SparseMatrixCOO convertToSparseStruct(const vector<vector<double>>& matrix);

/**
 * @brief Converts the SparseMatrixCOO structure back to a dense matrix.
 * @param sparse The sparse matrix.
 * @return vector<vector<double>> The dense matrix representation.
 */
vector<vector<double>> convertToDense(const SparseMatrixCOO& sparse);

// -------------------------------------------------------------------------
// Sparse Matrix Function Prototypes (Declarations)
// -------------------------------------------------------------------------

SparseMatrixCOO sparseZeroMatrix(int R, int C);

/**
 * @brief Creates a sparse identity matrix of a given size.
 * @param size The number of rows and columns for the square identity matrix.
 * @return SparseMatrixCOO The generated identity matrix.
 */
SparseMatrixCOO sparseIdentityMatrix(int size);

/**
 * @brief Concatenates two sparse matrices horizontally or vertically.
 * @param sparseM1 The primary matrix.
 * @param sparseM2 The secondary matrix to be appended.
 * @param side The direction of concatenation.
 * @return SparseMatrixCOO The new combined matrix.
 */
SparseMatrixCOO sparseConcatonate(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2, sparseConcatonateSide side);

/**
 * @brief Truncates a sparse matrix by removing the higher-indexed rows or columns.
 * @param sparseM The matrix to truncate.
 * @param indexCount The number of rows/columns to virtually remove.
 * @param indexType The dimension to truncate (ROW or COLUMN).
 * @return SparseMatrixCOO The truncated matrix.
 */
SparseMatrixCOO sparseTruncate(const SparseMatrixCOO& sparseM, int indexCount, sparseTruncateIndex indexType);

/**
 * @brief Computes the transpose of a sparse matrix.
 * @param sparseM The matrix to find the transpose of.
 * @return SparseMatrixCOO the transpose matrix.
 */
SparseMatrixCOO sparseTranspose(const SparseMatrixCOO& sparseM);

/**
 * @brief Computes the inverse matrix of a sparse matrix (via dense conversion and RREF).
 * @param sparseM The matrix to find the inverse of.
 * @return SparseMatrixCOO the inverse matrix.
 */
SparseMatrixCOO sparseInverse(const SparseMatrixCOO& sparseM);

/**
 * @brief Computes the determinant of a sparse matrix (via dense conversion).
 * @param sparseM The matrix to find the determinant of.
 * @return double the determinant of matrix.
 */
double sparseCalculateDeterminant(const SparseMatrixCOO& sparseM);

/**
 * @brief Displays a sparse matrix (by converting to dense).
 * @param sparseM The matrix to print.
 */
void sparsePrintMatrix(const SparseMatrixCOO& sparseM);

// -------------------------------------------------------------------------
// Operator Overloads
// -------------------------------------------------------------------------

/**
 * @brief Operator overload for sparse matrix multiplication (via dense conversion).
 */
SparseMatrixCOO operator*(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2);

/**
 * @brief Operator overload for sparse matrix addition (simple list merge - consolidation required for correctness).
 */
SparseMatrixCOO operator+(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2);

/**
 * @brief Operator overload for sparse matrix subtraction (simple list merge with negation - consolidation required for correctness).
 */
SparseMatrixCOO operator-(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2);

#endif // SPARSEMATRIX_H