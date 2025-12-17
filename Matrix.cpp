// File name : Matrix.cpp
// Last edited : 11/2/2025
// Author : Davis Lester
// Description : Creating Functions for Linear Algebra-based Calculations on 2D vectors (Matricies)

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
using namespace std;

//╭━╮╭━╮╱╱╭╮╱╱╱╱╱╱╱╭━━━╮╱╱╱╱╱╱╱╱╱╱╱╱╱╭╮╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╭╮╭╮╱╭╮╭╮╱╭╮╱╱╱╱╱╱╱╱╭╮
//┃┃╰╯┃┃╱╭╯╰╮╱╱╱╱╱╱┃╭━╮┃╱╱╱╱╱╱╱╱╱╱╱╱╭╯╰╮╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱┃┃┃┃╱┃┣╯╰╮┃┃╱╱╱╱╱╱╱╭╯╰╮
//┃╭╮╭╮┣━┻╮╭╋━┳┳╮╭╮┃┃╱╰╋━━┳━╮╭━━┳━┳━┻╮╭╋┳━━┳━╮╱╭━━┳━╮╭━╯┃┃┃╱┃┣╮╭╋┫┃╭┳━━━┳━┻╮╭╋┳━━┳━╮
//┃┃┃┃┃┃╭╮┃┃┃╭╋╋╋╋╯┃┃╭━┫┃━┫╭╮┫┃━┫╭┫╭╮┃┃┣┫╭╮┃╭╮╮┃╭╮┃╭╮┫╭╮┃┃┃╱┃┃┃┃┣┫┃┣╋━━┃┃╭╮┃┃┣┫╭╮┃╭╮╮
//┃┃┃┃┃┃╭╮┃╰┫┃┃┣╋╋╮┃╰┻━┃┃━┫┃┃┃┃━┫┃┃╭╮┃╰┫┃╰╯┃┃┃┃┃╭╮┃┃┃┃╰╯┃┃╰━╯┃┃╰┫┃╰┫┃┃━━┫╭╮┃╰┫┃╰╯┃┃┃┃
//╰╯╰╯╰┻╯╰┻━┻╯╰┻╯╰╯╰━━━┻━━┻╯╰┻━━┻╯╰╯╰┻━┻┻━━┻╯╰╯╰╯╰┻╯╰┻━━╯╰━━━╯╰━┻┻━┻┻━━━┻╯╰┻━┻┻━━┻╯╰╯

/**
 * @brief Creates an identity matrix of a given size.
 * * An identity matrix has ones on the main diagonal and zeros elsewhere.
 * * @param size The number of rows and columns for the square identity matrix.
 * @return std::vector<std::vector<int>> The generated identity matrix.
 */
vector<vector<double>> identityMatrix(int size) {

    // Create a square matrix
    vector<vector<double>> matrix(size, vector<double>(size));

    // Add a 1 down the main diagonal
    for (int i = 0; i < size; i++ ){
        matrix[i][i] = 1;
    }

    return matrix;

}

/**
 * @brief Creates a matrix full of 0s from a given size
 * * @param rows The number of rows for the matrix.
 * * @param columns The number of columns for the matrix.
 * @return std::vector<std::vector<int>> The generated identity matrix.
 */
vector<vector<double>> zeroMatrix(int rows, int columns) {

    vector<vector<double>> matrix(rows, vector<double>(columns));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = 0;
        }
    }

    return matrix;

}

// Creating an enumeration class to hold all the possible values for matrix concatonation
enum class concatonateSide { LEFT, RIGHT, TOP, BOTTOM };

/**
 * @brief Concatenates two matrices horizontally (LEFT/RIGHT) or vertically (TOP/BOTTOM).
 * * Throws an exception if the matrices are incompatible for the specified side.
 * * @param matrix1 The primary matrix, which retains its position.
 * @param matrix2 The secondary matrix to be appended.
 * @param side The direction of concatenation (LEFT, RIGHT, TOP, or BOTTOM).
 * @return std::vector<std::vector<int>> The new combined matrix.
 * @throw std::runtime_error if matrices have incompatible dimensions.
 */
vector<vector<double>> concatonate(vector<vector<double>> matrix1, vector<vector<double>> matrix2, concatonateSide side) {

    // matrix1.size() returns the number of rows
    // matrix1[0].size() returns the number of columns

    // Size of both matricies are 0
    if ((matrix1.size() == 0 || matrix1[0].size() == 0) && (matrix2.size() == 0 || matrix2[0].size() == 0)) {
        throw runtime_error("Both matricies entered have size 0");
    }
    // Size of matrix1 is 0, return matrix2
    else if (matrix1.size() == 0 || matrix1[0].size() == 0) {
        return matrix2;
    }
    // Size of matrix2 is 0, return matrix1
    else if (matrix2.size() == 0 || matrix2[0].size() == 0) {
        return matrix1;
    }

    // Code for adding the matrix to the side
    if ((side == concatonateSide::LEFT) || (side == concatonateSide::RIGHT) ) {
        if (matrix1.size() != matrix2.size()) { 
            throw runtime_error("Rows must be equal for horizontal concatonation"); 
        }
        // Add number of columns to add the matrix to the left or the right
        vector<vector<double>> matrix(matrix1.size(), vector<double>(matrix1[0].size() + matrix2[0].size()));

        // Loop through rows
        for (int r = 0; r < matrix1.size(); r++) {
            // Copy matrix1
            // Loop through columns of matrix1
            for (int c = 0; c < matrix1[0].size(); c++) {
                // matrix1 goes to the right for LEFT side, and to the left for RIGHT side
                int target_c = (side == concatonateSide::RIGHT) ? c : matrix2[0].size() + c;
                matrix[r][target_c] = matrix1[r][c];
            }
            
            // Copy matrix2
            for (int c = 0; c < matrix2[0].size(); c++) { // Loop through columns of matrix2
                // matrix2 goes to the left for LEFT side, and to the right for RIGHT side
                int target_c = (side == concatonateSide::LEFT) ? c : matrix1[0].size() + c;
                matrix[r][target_c] = matrix2[r][c];
            }
        }

        return matrix;

    }

    // Code for adding the matrix on top or on bottom
    if ((side == concatonateSide::TOP) || (side == concatonateSide::BOTTOM) ) {
        if (matrix1[0].size() != matrix2[0].size()) { 
            throw runtime_error("Columns must be equal for vertical concatonation"); 
        }
        // Add number of rows to add the matrix to the top or bottom
        vector<vector<double>> matrix(matrix1.size() + matrix2.size(), vector<double>(matrix1[0].size()));

        // Loop through columns
        for (int c = 0; c < matrix1[0].size(); c++) {
            
            // Copy matrix1
            // Loop through rows of matrix1
            for (int r = 0; r < matrix1.size(); r++) {
                // matrix1 goes to the bottom for TOP side, and to the top for BOTTOM side
                int target_r = (side == concatonateSide::BOTTOM) ? r : matrix2.size() + r;
                matrix[target_r][c] = matrix1[r][c];
            }

            // Copy matrix2
            // Loop through rows of matrix2
            for (int r = 0; r < matrix2.size(); r++) {
                // matrix2 goes to the top for TOP side, and to the bottom for BOTTOM side
                int target_r = (side == concatonateSide::TOP) ? r : matrix1.size() + r;
                matrix[target_r][c] = matrix2[r][c];
            }
        }

        return matrix;

    }

    throw runtime_error("Invalid concatonation type specified."); 

}

// Creating an enumeration class to hold all the possible values for matrix truncation
enum class truncateIndex { ROW, COLUMN };

vector<vector<double>> truncate(const vector<vector<double>>& matrix, int index, truncateIndex indexType) {

    // matrix1.size() returns the number of rows
    // matrix1[0].size() returns the number of columns

    if (matrix.size() == 0 || matrix[0].size() == 0) {
        throw runtime_error("Cannot truncate a matrix of 0 size"); 
    }

    if (index < 0) {
        throw runtime_error("The number of items to remove cannot be negative.");
    }

    if (indexType == truncateIndex::ROW) {

        if (index >= matrix.size()) {
            throw runtime_error("Cannot remove more than the entire matrix");
        }

        vector<vector<double>> truncatedMatrix(matrix.size() - index, vector<double>(matrix[0].size()));

        for(int i = 0; i < matrix.size() - index; i++) {
            for(int j = 0; j < matrix[0].size(); j++) {
                truncatedMatrix[i][j] = matrix[i][j];
            }
        }

        return truncatedMatrix;

    }

    if (indexType == truncateIndex::COLUMN) {

        if (index >= matrix[0].size()) {
            throw runtime_error("Cannot remove more than the entire matrix");
        }

        vector<vector<double>> truncatedMatrix(matrix.size(), vector<double>(matrix[0].size() - index));

        for(int i = 0; i < matrix.size(); i++) {
            for(int j = 0; j < matrix[0].size() - index; j++) {
                truncatedMatrix[i][j] = matrix[i][j];
            }
        }

        return truncatedMatrix;

    }

    throw runtime_error("Invalid truncation index type specified."); 

}

/**
 * @brief Solves a system of linar equations represented as a matrix.
 * * Utilizes Gaussian Elimination for Reduced Row Echelon Form
 * * @param matrix The matrix which needs to be solved.
 * @return std::vector<std::vector<int>> The solved matrix.
 * @throw std::runtime_error if matrix has 0 rows.
 */
vector<vector<double>> RREF(vector<vector<double>> matrix) {

    // matrix1.size() returns the number of rows
    // matrix1[0].size() returns the number of columns

    if (matrix.size() == 0) {
        throw runtime_error("Cannot RREF a matrix with 0 rows");
    }

    int lead = 0; // Current leading column index
    for (int r = 0; r < matrix.size() && lead < matrix[0].size(); ++r) {
        int i = r;
        // Find a pivot row (row with a non-zero entry in the current leading column)
        while (i < matrix.size() && matrix[i][lead] == 0) {
            i++;
        }

        // If a pivot row is found
        if (i < matrix.size()) {
            // Swap current row with the pivot row
            swap(matrix[r], matrix[i]);

            // Scale the pivot row to make the pivot element 1
            double div = matrix[r][lead];
            for (int j = lead; j < matrix[0].size(); ++j) {
                matrix[r][j] /= div;
            }

            // Eliminate other entries in the pivot column
            for (int i_other = 0; i_other < matrix.size(); ++i_other) {
                if (i_other != r) {
                    double mult = matrix[i_other][lead];
                    for (int j = lead; j < matrix[0].size(); ++j) {
                        matrix[i_other][j] -= mult * matrix[r][j];
                    }
                }
            }
            lead++; // Move to the next leading column
        } else { // No pivot found in this column, move to the next column
            lead++;
            r--; // Re-process the current row with the new leading column
        }
    }

    return matrix;

}

/**
 * @brief Removes a row and column from a larger matrix.
 * * @param matrix The matrix to have a row and column removed.
 * * @param rowToRemove The row index to be removed from the matrix.
 * * @param coltoRemove The column index to be removed from the matrix.
 * @return vector<vector<double>> The original matrix with the row and column removed.
 */
vector<vector<double>> getSubmatrix(const vector<vector<double>>& matrix, int rowToRemove, int colToRemove) {
    int n = matrix.size();
    // Create a new matrix with 1 less column and row
    vector<vector<double>> submatrix(n - 1, vector<double>(n - 1));

    int sub_row = 0;
    // Search for the row to remove
    for (int i = 0; i < n; ++i) {
        if (i == rowToRemove) continue;
        
        int sub_col = 0;
        // Search for the column to remove
        for (int j = 0; j < n; ++j) {
            if (j == colToRemove) continue;
            
            // Assign data to the new matrix
            submatrix[sub_row][sub_col] = matrix[i][j];
            sub_col++;
        }
        sub_row++;
    }
    return submatrix;
}

/**
 * @brief Computes the determinant of a square matrix.
 * * Utilizes the Cofactor Expansion Algorithm.
 * * @param matrix The matrix to find the determinant of.
 * @return double the determinant of matrix.
 * @throw std::runtime_error if matrix is not square.
 */
double calculateDeterminant(vector<vector<double>> matrix) {

    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Cannot find the determinant of a non-square matrix
    if (matrix.size() != matrix[0].size()) {
        throw runtime_error("Cannot find the determinant of a non-square matrix");
    }

    // Determinant of a 1x1 matrix is the only value in it
    if ((matrix.size() == 1) && (matrix[0].size() == 1)) {
        return matrix[0][0];
    }

    // Determinant of a 2x2 matrix is easy to compute
    if ((matrix.size() == 2) && (matrix[0].size() == 2)) {
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]));
    }

    double determinant = 0.0;
    int sign = 1; // For alternating signs (+1, -1, +1, ...)

    // Iterate through the first row (or any row/column)
    for (int j = 0; j < matrix.size(); ++j) {

        // Get the submatrix by removing the first row and current column 'j'
        vector<vector<double>> submatrix = getSubmatrix(matrix, 0, j);

        // Calculate the cofactor and add to the determinant
        determinant += sign * matrix[0][j] * calculateDeterminant(submatrix);

        // Alternate the sign for the next element
        sign *= -1;
    }

    return determinant;

}

/**
 * @brief Computes the transpose of a matrix.
 * * @param matrix The matrix to find the transpose of.
 * @return vector<vector<double> the transpose matrix.
 * @throw std::runtime_error if matrix is empty.
 */
vector<vector<double>> transpose(vector<vector<double>> matrix) {

    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Trivial solution, no transpose of an empty matrix
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        throw runtime_error("Cannot find the transpose of an empty matrix");
    }

    // Create new transpose matrix
    vector<vector<double>> transposeMatrix(matrix[0].size(), vector<double>(matrix.size()));

    // Iterate through all values and swap them.
    // Note: Data is not lost since data is swapped to a new matrix
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            transposeMatrix[j][i] =  matrix[i][j];
        }
    }

    return transposeMatrix;

}

/**
 * @brief Computes the inverse matrix of a matrix.
 * * @param matrix The matrix to find the inverse of.
 * @return vector<vector<double> the inverse matrix.
 * @throw std::runtime_error if matrix is not square or empty.
 */
vector<vector<double>> inverse(vector<vector<double>> matrix) {

    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Cannot inverse a non-square matrix
    if (matrix.size() != matrix[0].size()) {
        throw runtime_error("Cannot invert a non-square matrix.");
    }

    // Cannot inverse a non-square matrix
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        throw runtime_error("Cannot invert an empty matrix.");
    }

    // Utilize A|I -> I|A^-1
    // 1. Create a square identity matrix of equal size to the passed matrix
    // 2. Combine the given matrix A on the left and the identity matrix on the right
    // 3. RREF the new combined matrix, resulting with the identity matrix on the left and the inverse matrix on the right
    // 4. Remove the Identity matrix and return the inverse matrix
    return (truncate(RREF(concatonate(matrix, identityMatrix(matrix.size()), concatonateSide::RIGHT)), matrix.size(), truncateIndex::COLUMN));

}

/**
 * @brief Displays a matrix to the terminal.
 * * @param matrix The matrix to print.
 * @throw std::runtime_error if matrix is empty.
 */
void printMatrix(vector<vector<double>> matrix) {

    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Cannot print an empty matrix
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        throw runtime_error("Cannot print an empty matrix.");
    }

    for(int i = 0; i < matrix.size(); i++) {
        for(int j = 0; j < matrix[0].size(); j++) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }

}

// Function to overload the * operator for matrix multiplication
vector<vector<double>> operator*(const vector<vector<double>> matrix1, const vector<vector<double>> matrix2) {
    
    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Check for compatible dimensions
    if ((matrix1[0].size() != matrix2.size())) {
        throw runtime_error("Matrices must have the same columns and rows for multiplication.");
    }

    // Initialize the result matrix
    vector<vector<double>> result(matrix1.size(), vector<double>(matrix2[0].size()));

    for (int i = 0; i < matrix1.size(); ++i) { 
        
        // Loop over columns of the result matrix
        for (int j = 0; j < matrix2[0].size(); ++j) {
            
            // Loop for the dot product
            double sum = 0.0;
            for (int k = 0; k < matrix1[0].size(); ++k) {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

/**
 * @brief Overloads the * operator for scalar multiplication (matrix * double).
 * @param matrix The matrix.
 * @param num The scalar value.
 * @return vector<vector<double>> The result of the multiplication.
 */
vector<vector<double>> operator*(const vector<vector<double>> matrix, const double num) {
    
    if (matrix.empty() || matrix[0].empty()) {
        throw runtime_error("Cannot multiply an empty matrix by a scalar.");
    }

    vector<vector<double>> result = matrix; // Start with a copy

    for (int i = 0; i < result.size(); ++i) {
        for (int j = 0; j < result[0].size(); ++j) {
            result[i][j] *= num; // Multiply each element by the scalar
        }
    }

    return result;
}

// Function to overload the + operator for matrix addition
vector<vector<double>> operator+(const vector<vector<double>> matrix1, const vector<vector<double>> matrix2) {
    
    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Check for compatible dimensions
    if ((matrix1.size() != matrix2.size()) || (matrix1[0].size() != matrix2[0].size())) {
        throw runtime_error("Matrices must have the same dimensions for addition.");
    }

    // Initialize the result matrix
    vector<vector<double>> result(matrix1.size(), vector<double>(matrix1[0].size()));

    // Perform element-wise addition
    for (int i = 0; i < matrix1.size(); ++i) {
        for (int j = 0; j < matrix1[0].size(); ++j) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return result;
}

// Function to overload the - operator for matrix subtraction
vector<vector<double>> operator-(const vector<vector<double>> matrix1, const vector<vector<double>> matrix2) {
    
    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // Check for compatible dimensions
    if ((matrix1.size() != matrix2.size()) || (matrix1[0].size() != matrix2[0].size())) {
        throw runtime_error("Matrices must have the same dimensions for subtraction.");
    }

    // Initialize the result matrix
    vector<vector<double>> result(matrix1.size(), vector<double>(matrix1[0].size()));

    // Perform element-wise subtraction
    for (int i = 0; i < matrix1.size(); ++i) {
        for (int j = 0; j < matrix1[0].size(); ++j) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    return result;
}

/**
 * @brief Displays a matrix to the terminal.
 * * @param matrix The matrix to print.
 * @return vector<vector<double> the sparse matrix.
 * @throw std::runtime_error if matrix is empty.
 */
vector<vector<double>> convertToSparse(vector<vector<double>> matrix) {

    // matrix.size() returns the number of rows
    // matrix[0].size() returns the number of columns

    // A sparse matrix is one where there are 3 rows, row 1 represents the X coordinate, row 2 represents the Y coordinate
    // and row 3 represents the value. Only values that are nonzero are stored in sparse matricies.

    // Cannot convert an empty matrix
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        throw runtime_error("Cannot convert an empty matrix into a sparse matrix.");
    }

    vector<vector<double>> sparse(3, vector<double>(0));

    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            if (matrix[i][j] != 0) {

                sparse[0].push_back(i);
                sparse[1].push_back(j);
                sparse[2].push_back(matrix[i][j]);

            }
        }
    }

    return sparse;

}