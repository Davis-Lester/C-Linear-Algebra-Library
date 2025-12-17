// File name : SparseMatrix.cpp
// Last edited : 11/2/2025
// Author : Davis Lester
// Description : Creating Functions for Linear Algebra-based Calculations on 2D vectors representing sparse matricies
// A sparse matrix is one where there are 3 rows, row 1 represents the X coordinate, row 2 represents the Y coordinate
// and row 3 represents the value. Only values that are nonzero are stored in sparse matricies.

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "Matrix.hpp"
using namespace std;

//╭━━━╮╱╱╱╱╱╱╱╱╱╱╱╱╱╱╭━╮╭━╮╱╱╭╮╱╱╱╱╱╱╱╭━━━╮╱╱╱╱╱╱╱╭╮╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╭╮╭╮╱╭╮╭╮╱╭╮╱╱╱╱╱╱╱╱╭╮
//┃╭━╮┃╱╱╱╱╱╱╱╱╱╱╱╱╱╱┃┃╰╯┃┃╱╭╯╰╮╱╱╱╱╱╱┃╭━╮┃╱╱╱╱╱╱╭╯╰╮╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱┃┃┃┃╱┃┣╯╰╮┃┃╱╱╱╱╱╱╱╭╯╰╮
//┃╰━━┳━━┳━━┳━┳━━┳━━╮┃╭╮╭╮┣━┻╮╭╋━┳┳╮╭╮┃┃╱╰╋━┳━━┳━┻╮╭╋┳━━┳━╮╱╭━━┳━╮╭━╯┃┃┃╱┃┣╮╭╋┫┃╭┳━━━┳━┻╮╭╋┳━━┳━╮
//╰━━╮┃╭╮┃╭╮┃╭┫━━┫┃━┫┃┃┃┃┃┃╭╮┃┃┃╭╋╋╋╋╯┃┃╱╭┫╭┫┃━┫╭╮┃┃┣┫╭╮┃╭╮╮┃╭╮┃╭╮┫╭╮┃┃┃╱┃┃┃┃┣┫┃┣╋━━┃┃╭╮┃┃┣┫╭╮┃╭╮╮
//┃╰━╯┃╰╯┃╭╮┃┃┣━━┃┃━┫┃┃┃┃┃┃╭╮┃╰┫┃┃┣╋╋╮┃╰━╯┃┃┃┃━┫╭╮┃╰┫┃╰╯┃┃┃┃┃╭╮┃┃┃┃╰╯┃┃╰━╯┃┃╰┫┃╰┫┃┃━━┫╭╮┃╰┫┃╰╯┃┃┃┃
//╰━━━┫╭━┻╯╰┻╯╰━━┻━━╯╰╯╰╯╰┻╯╰┻━┻╯╰┻╯╰╯╰━━━┻╯╰━━┻╯╰┻━┻┻━━┻╯╰╯╰╯╰┻╯╰┻━━╯╰━━━╯╰━┻┻━┻┻━━━┻╯╰┻━┻┻━━┻╯╰╯
//╱╱╱╱┃┃
//╱╱╱╱╰╯

// The essential structure for sparse matrix handling. It stores the full dimensions (R x C)
// alongside the Coordinate List (COO) data.
struct SparseMatrixCOO {
    size_t R; // Full number of rows
    size_t C; // Full number of columns
    std::vector<double> row_indices; // Row 0 in your old format
    std::vector<double> col_indices; // Row 1 in your old format
    std::vector<double> values;      // Row 2 in your old format
};

// Creating an enumeration class to hold all the possible values for matrix concatonation
enum class sparseConcatonateSide { LEFT, RIGHT, TOP, BOTTOM };

// Creating an enumeration class to hold all the possible values for matrix truncation
enum class sparseTruncateIndex { ROW, COLUMN };

/**
 * @brief Converts a dense matrix to the SparseMatrixCOO structure.
 */
SparseMatrixCOO convertToSparseStruct(const vector<vector<double>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        // Correctly handle the case of an empty dense matrix by 
        // returning a SparseMatrixCOO with R=0 and C=0.
        // This prevents the runtime error.
        if (matrix.empty()) {
            return {0, 0, {}, {}, {}};
        } else { // matrix is not empty, but has 0 columns (i.e., matrix[0].empty())
            return {matrix.size(), 0, {}, {}, {}};
        }
    }

    SparseMatrixCOO sparseM;
    sparseM.R = matrix.size();
    sparseM.C = matrix[0].size();

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            if (matrix[i][j] != 0.0) {
                sparseM.row_indices.push_back(i);
                sparseM.col_indices.push_back(j);
                sparseM.values.push_back(matrix[i][j]);
            }
        }
    }
    return sparseM;
}

/**
 * @brief Converts the SparseMatrixCOO structure back to a dense matrix.
 */
vector<vector<double>> convertToDense(const SparseMatrixCOO& sparse) {
    vector<vector<double>> dense(sparse.R, vector<double>(sparse.C, 0.0));
    for (size_t k = 0; k < sparse.values.size(); ++k) {
        // Indices are stored as doubles but should be safely cast to size_t (assuming non-negative integers)
        size_t r = static_cast<size_t>(sparse.row_indices[k]);
        size_t c = static_cast<size_t>(sparse.col_indices[k]);
        if (r < sparse.R && c < sparse.C) {
            dense[r][c] = sparse.values[k];
        }
    }
    return dense;
}

// -----------------------------------------------------------------------------------------
// Converted Sparse Functions
// -----------------------------------------------------------------------------------------

SparseMatrixCOO sparseZeroMatrix(int R, int C) {
    return { (size_t)R, (size_t)C, {}, {}, {} };
}

/**
 * @brief Creates a sparse identity matrix of a given size.
 * @param size The number of rows and columns for the square identity matrix.
 * @return SparseMatrixCOO The generated identity matrix.
 */
SparseMatrixCOO sparseIdentityMatrix(int size) {
    if (size <= 0) {
        return {0, 0, {}, {}, {}};
    }

    SparseMatrixCOO sparse;
    sparse.R = size;
    sparse.C = size;
    
    // An NxN identity matrix has N non-zero entries on the diagonal
    for (int i = 0; i < size; i++ ){
        sparse.row_indices.push_back(i);
        sparse.col_indices.push_back(i);
        sparse.values.push_back(1.0);
    }
    return sparse;
}

/**
 * @brief Concatenates two sparse matrices horizontally (LEFT/RIGHT) or vertically (TOP/BOTTOM).
 * @param sparseM1 The primary matrix.
 * @param sparseM2 The secondary matrix to be appended.
 * @param side The direction of concatenation.
 * @return SparseMatrixCOO The new combined matrix.
 */
SparseMatrixCOO sparseConcatonate(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2, sparseConcatonateSide side) {
    
    // Handle empty matrices
    bool m1_empty = (sparseM1.R == 0 || sparseM1.C == 0);
    bool m2_empty = (sparseM2.R == 0 || sparseM2.C == 0);

    if (m1_empty && m2_empty) {
        throw runtime_error("Both matrices entered have size 0");
    } else if (m1_empty) {
        return sparseM2;
    } else if (m2_empty) {
        return sparseM1;
    }

    SparseMatrixCOO sparseResult;

    // Horizontal Concatenation (LEFT/RIGHT) ---
    if (side == sparseConcatonateSide::LEFT || side == sparseConcatonateSide::RIGHT) {
        if (sparseM1.R != sparseM2.R) { 
            throw runtime_error("Row dimensions must match for horizontal concatenation."); 
        }
        
        sparseResult.R = sparseM1.R;
        sparseResult.C = sparseM1.C + sparseM2.C;

        // Determine which matrix is on the left to set the shift amount
        const SparseMatrixCOO& M_left = (side == sparseConcatonateSide::RIGHT) ? sparseM1 : sparseM2;
        const SparseMatrixCOO& M_right = (side == sparseConcatonateSide::RIGHT) ? sparseM2 : sparseM1;
        const size_t col_shift = M_left.C;

        // Copy M_left (no shift needed)
        sparseResult.row_indices.insert(sparseResult.row_indices.end(), M_left.row_indices.begin(), M_left.row_indices.end());
        sparseResult.col_indices.insert(sparseResult.col_indices.end(), M_left.col_indices.begin(), M_left.col_indices.end());
        sparseResult.values.insert(sparseResult.values.end(), M_left.values.begin(), M_left.values.end());

        // Copy M_right (column shift required)
        sparseResult.row_indices.insert(sparseResult.row_indices.end(), M_right.row_indices.begin(), M_right.row_indices.end());
        for (double col_idx : M_right.col_indices) {
            sparseResult.col_indices.push_back(col_idx + col_shift);
        }
        sparseResult.values.insert(sparseResult.values.end(), M_right.values.begin(), M_right.values.end());
    }
    
    // Vertical Concatenation (TOP/BOTTOM) ---
    else if (side == sparseConcatonateSide::TOP || side == sparseConcatonateSide::BOTTOM) {
        if (sparseM1.C != sparseM2.C) { 
            throw runtime_error("Column dimensions must match for vertical concatenation."); 
        }

        sparseResult.R = sparseM1.R + sparseM2.R;
        sparseResult.C = sparseM1.C;

        // Determine which matrix is on top to set the shift amount
        const SparseMatrixCOO& M_top = (side == sparseConcatonateSide::BOTTOM) ? sparseM1 : sparseM2;
        const SparseMatrixCOO& M_bottom = (side == sparseConcatonateSide::BOTTOM) ? sparseM2 : sparseM1;
        const size_t row_shift = M_top.R;

        // Copy M_top (no shift needed)
        sparseResult.row_indices.insert(sparseResult.row_indices.end(), M_top.row_indices.begin(), M_top.row_indices.end());
        sparseResult.col_indices.insert(sparseResult.col_indices.end(), M_top.col_indices.begin(), M_top.col_indices.end());
        sparseResult.values.insert(sparseResult.values.end(), M_top.values.begin(), M_top.values.end());

        // Copy M_bottom (row shift required)
        for (double row_idx : M_bottom.row_indices) {
            sparseResult.row_indices.push_back(row_idx + row_shift);
        }
        sparseResult.col_indices.insert(sparseResult.col_indices.end(), M_bottom.col_indices.begin(), M_bottom.col_indices.end());
        sparseResult.values.insert(sparseResult.values.end(), M_bottom.values.begin(), M_bottom.values.end());
    } else {
        throw runtime_error("Invalid concatenation side specified."); 
    }

    return sparseResult;
}

/**
 * @brief Truncates a sparse matrix by removing the higher-indexed rows or columns.
 * @param sparseM The matrix to truncate.
 * @param indexCount The number of rows/columns to virtually remove.
 * @param indexType The dimension to truncate (ROW or COLUMN).
 * @return SparseMatrixCOO The truncated matrix.
 */
SparseMatrixCOO sparseTruncate(const SparseMatrixCOO& sparseM, int indexCount, sparseTruncateIndex indexType) {
    if (sparseM.R == 0 || sparseM.C == 0) {
        throw runtime_error("Cannot truncate a matrix of 0 size."); 
    }
    if (indexCount < 0) {
        throw runtime_error("The number of items to remove cannot be negative.");
    }
    
    SparseMatrixCOO truncated = {sparseM.R, sparseM.C, {}, {}, {}};
    size_t new_R = sparseM.R;
    size_t new_C = sparseM.C;

    if (indexType == sparseTruncateIndex::ROW) {
        if ((size_t)indexCount >= sparseM.R) {
            throw runtime_error("Cannot remove more rows than the matrix has.");
        }
        new_R = sparseM.R - indexCount;
    } else if (indexType == sparseTruncateIndex::COLUMN) {
        if ((size_t)indexCount >= sparseM.C) {
            throw runtime_error("Cannot remove more columns than the matrix has.");
        }
        new_C = sparseM.C - indexCount;
    } else {
        throw runtime_error("Invalid truncation index type specified."); 
    }
    
    truncated.R = new_R;
    truncated.C = new_C;

    // Only keep elements whose indices are within the new bounds [0, new_R) and [0, new_C)
    for (size_t k = 0; k < sparseM.values.size(); ++k) {
        size_t r = static_cast<size_t>(sparseM.row_indices[k]);
        size_t c = static_cast<size_t>(sparseM.col_indices[k]);
        
        bool keep = (r < new_R) && (c < new_C);
        
        if (keep) {
            truncated.row_indices.push_back(r);
            truncated.col_indices.push_back(c);
            truncated.values.push_back(sparseM.values[k]);
        }
    }
    return truncated;
}

/**
 * @brief Computes the transpose of a sparse matrix.
 * @param sparseM The matrix to find the transpose of.
 * @return SparseMatrixCOO the transpose matrix.
 */
SparseMatrixCOO sparseTranspose(const SparseMatrixCOO& sparseM) {
    // Check if the matrix is empty
    if (sparseM.R == 0 || sparseM.C == 0) {
        // Return an empty sparse matrix with swapped (but still zero) dimensions
        return {sparseM.C, sparseM.R, {}, {}, {}};
    }
    
    SparseMatrixCOO result;
    // Swap the dimensions
    result.R = sparseM.C;
    result.C = sparseM.R;
    
    // Swap the row and column indices
    result.row_indices = sparseM.col_indices;
    result.col_indices = sparseM.row_indices;
    result.values = sparseM.values; // Values remain the same
    
    return result;
}

/**
 * @brief Computes the inverse matrix of a sparse matrix.
 * NOTE: Requires conversion to dense for Gaussian Elimination (RREF).
 * @param sparseM The matrix to find the inverse of.
 * @return SparseMatrixCOO the inverse matrix.
 */
SparseMatrixCOO sparseInverse(const SparseMatrixCOO& sparseM) {
    if (sparseM.R != sparseM.C) {
        throw runtime_error("Cannot invert a non-square matrix.");
    }
    if (sparseM.R == 0) {
        throw runtime_error("Cannot invert an empty matrix.");
    }
    
    // Convert A to dense and create dense Identity I
    vector<vector<double>> denseA = convertToDense(sparseM);
    vector<vector<double>> identityI(sparseM.R, vector<double>(sparseM.R, 0.0));
    for(size_t i = 0; i < sparseM.R; ++i) identityI[i][i] = 1.0;

    // Concatenate to [A | I]
    vector<vector<double>> augmented(sparseM.R, vector<double>(sparseM.C * 2));
    for(size_t i = 0; i < sparseM.R; ++i) {
        copy(denseA[i].begin(), denseA[i].end(), augmented[i].begin());
        copy(identityI[i].begin(), identityI[i].end(), augmented[i].begin() + sparseM.C);
    }
    
    // RREF the augmented matrix
    vector<vector<double>> rrefAugmented = RREF(augmented); 

    // Truncate the right side to get A^-1
    vector<vector<double>> denseInverse(sparseM.R, vector<double>(sparseM.C));
    for (size_t i = 0; i < sparseM.R; ++i) {
        copy(rrefAugmented[i].begin() + sparseM.C, rrefAugmented[i].end(), denseInverse[i].begin());
    }
    
    // Convert A^-1 back to sparse
    return convertToSparseStruct(denseInverse);
}

/**
 * @brief Computes the determinant of a sparse matrix.
 * NOTE: Requires conversion to dense for Cofactor Expansion.
 * @param sparseM The matrix to find the determinant of.
 * @return double the determinant of matrix.
 */
double sparseCalculateDeterminant(const SparseMatrixCOO& sparseM) {
    if (sparseM.R != sparseM.C) {
        throw runtime_error("Cannot find the determinant of a non-square matrix");
    }
    
    // Convert to dense to perform the calculation
    vector<vector<double>> denseM = convertToDense(sparseM);
    return calculateDeterminant(denseM);
}

/**
 * @brief Displays a sparse matrix (by converting to dense).
 * @param sparseM The matrix to print.
 */
void sparsePrintMatrix(const SparseMatrixCOO& sparseM) {
    if (sparseM.R == 0 || sparseM.C == 0) {
        throw runtime_error("Cannot print an empty matrix.");
    }
    
    // Convert to dense for readable display
    vector<vector<double>> denseM = convertToDense(sparseM);

    for(size_t i = 0; i < sparseM.R; i++) {
        for(size_t j = 0; j < sparseM.C; j++) {
            cout << denseM[i][j] << "\t";
        }
        cout << endl;
    }
}

/**
 * @brief Operator overload for sparse matrix multiplication.
 * NOTE: Requires conversion to dense for computation.
 */
SparseMatrixCOO operator*(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2) {
    if (sparseM1.C != sparseM2.R) {
        throw runtime_error("Matrices must have the same inner dimensions (C1 == R2) for multiplication.");
    }
    
    // Convert to dense to perform multiplication
    vector<vector<double>> denseM1 = convertToDense(sparseM1);
    vector<vector<double>> denseM2 = convertToDense(sparseM2);
    
    // Perform multiplication using dense matrix logic
    vector<vector<double>> resultDense(denseM1.size(), vector<double>(denseM2[0].size()));

    for (size_t i = 0; i < denseM1.size(); ++i) { 
        for (size_t j = 0; j < denseM2[0].size(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < denseM1[0].size(); ++k) {
                sum += denseM1[i][k] * denseM2[k][j];
            }
            resultDense[i][j] = sum;
        }
    }
    
    // Convert back to sparse
    return convertToSparseStruct(resultDense);
}

/**
 * @brief Operator overload for sparse matrix addition.
 * NOTE: Efficient sparse addition requires handling duplicates (summing values at the same (R, C)).
 * This implementation simply merges the lists, which may not be mathematically correct without a final consolidation step.
 */
SparseMatrixCOO operator+(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2) {
    if ((sparseM1.R != sparseM2.R) || (sparseM1.C != sparseM2.C)) {
        throw runtime_error("Matrices must have the same dimensions for addition.");
    }
    
    SparseMatrixCOO result = {sparseM1.R, sparseM1.C, {}, {}, {}};
    
    // Simple list concatenation
    result.row_indices.insert(result.row_indices.end(), sparseM1.row_indices.begin(), sparseM1.row_indices.end());
    result.col_indices.insert(result.col_indices.end(), sparseM1.col_indices.begin(), sparseM1.col_indices.end());
    result.values.insert(result.values.end(), sparseM1.values.begin(), sparseM1.values.end());
    
    result.row_indices.insert(result.row_indices.end(), sparseM2.row_indices.begin(), sparseM2.row_indices.end());
    result.col_indices.insert(result.col_indices.end(), sparseM2.col_indices.begin(), sparseM2.col_indices.end());
    result.values.insert(result.values.end(), sparseM2.values.begin(), sparseM2.values.end());
    
    return result;
}

/**
 * @brief Operator overload for sparse matrix subtraction.
 */
SparseMatrixCOO operator-(const SparseMatrixCOO& sparseM1, const SparseMatrixCOO& sparseM2) {
    if ((sparseM1.R != sparseM2.R) || (sparseM1.C != sparseM2.C)) {
        throw runtime_error("Matrices must have the same dimensions for subtraction.");
    }
    
    SparseMatrixCOO result = {sparseM1.R, sparseM1.C, {}, {}, {}};
    
    // Copy all elements from M1
    result.row_indices.insert(result.row_indices.end(), sparseM1.row_indices.begin(), sparseM1.row_indices.end());
    result.col_indices.insert(result.col_indices.end(), sparseM1.col_indices.begin(), sparseM1.col_indices.end());
    result.values.insert(result.values.end(), sparseM1.values.begin(), sparseM1.values.end());
    
    // Copy all elements from M2, negating the value (M1 + (-M2))
    result.row_indices.insert(result.row_indices.end(), sparseM2.row_indices.begin(), sparseM2.row_indices.end());
    result.col_indices.insert(result.col_indices.end(), sparseM2.col_indices.begin(), sparseM2.col_indices.end());
    for (double val : sparseM2.values) {
        result.values.push_back(-val);
    }
    
    return result;
}