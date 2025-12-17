<div align="center">

# Linear Algebra Library
### *Advanced Linear Algebra in C++*

![C++17](https://img.shields.io/badge/Language-C%2B%2B17-00599C?style=flat&logo=cplusplus&logoColor=white)
![UF Engineering](https://img.shields.io/badge/University%20of%20Florida-EE-FA4616?style=flat&logo=university-of-florida&logoColor=white)
![Status](https://img.shields.io/badge/Project-Linear%20Algebra-blueviolet?style=flat)

**Developed by Davis Lester**
*Sophomore Electrical Engineering Student & Peer Advisor @ University of Florida*

---

### 🚀 Overview
This linear algebra library is a lightweight C++ library for 2D matrix operations. Designed for solving voltage equations, this library makes linear algebra intuitive and fast for both dense and sparse matrices.

</div>

---

### ✨ Features
1. **Identity and Zero Matrices**: Quick generation of standard matrices.
2. **Matrix Concatenation**: Combine matrices via `LEFT`, `RIGHT`, `TOP`, or `BOTTOM`.
3. **Truncation**: Effortlessly remove specific rows or columns.
4. **Reduced Row Echelon Form (RREF)**: Solve systems of linear equations.
5. **Determinant Calculation**: Efficiently compute matrix determinants.
6. **Transpose & Inverse**: Essential transformations for circuit analysis.
7. **Sparse Matrix Conversion**: Optimize memory for large-scale systems.
8. **Operator Overloads**: Natural syntax using `+`, `-`, and `*`.
9. **Robust Error Handling**: Built-in validation for matrix dimensions.

---

### ⚡ Dense vs. Sparse Matrices


* **Dense Matrix**: All elements are stored (standard $m \times n$ storage).
* **Sparse Matrix**: Only non-zero elements are stored as `[row, col, value]`.

Sparse matrices save memory when most elements are zero.



```cpp
#define USE_SPARSE_MATRIX
#include "Matrix.hpp"```

### 📦 Installation
* Include ```Matrix.hpp``` and its header in your project. No external dependencies required. Works on any standard C++17 compiler.

```cpp
#include "Matrix.hpp"
using namespace std;```

### 💻 Example Usage
```cpp
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
}```

<div align="center">

</div>