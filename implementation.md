# README

## Updates Overview

This document provides an overview of the updates made to the `layout.csl` and `pe_program.csl` files as HW1.0, HW1.1, HW2.0, and HW2.1 of the project.

### 1. Checkerboard Color Assignment in `layout.csl`

The `layout.csl` file has been updated to include a new implementation for assigning colors to a checkerboard layout. The key changes are as follows:
- **Algorithm Update**: Implemented in lines `45-78`, the algorithm alternates colors across rows and columns using a nested loop structure.
- **Efficiency Improvements**: Lines `60-65` optimize the logic by caching intermediate results to minimize redundant computations.
- **Code Modularity**: Functions `assign_checkerboard_colors` (lines `30-44`) and `validate_checkerboard` (lines `80-95`) were added to improve code reuse and maintainability.

### 2. `col_sort` and `row_sort` Algorithms in `pe_program.csl`

The `pe_program.csl` file now includes implementations for the `col_sort` and `row_sort` algorithms. These algorithms are designed to handle parallel sorting operations efficiently.

#### `col_sort` Algorithm:
- **Functionality**: Defined in lines `120-150`, this function sorts elements column-wise in a 2D grid.
- **Parallelism**: Lines `130-140` utilize OpenMP directives to enable parallel processing for improved performance.
- **Edge Cases**: Lines `145-150` include logic to handle uneven column lengths.

#### `row_sort` Algorithm:
- **Functionality**: Implemented in lines `160-190`, this function sorts elements row-wise in a 2D grid.
- **Parallelism**: Lines `170-180` leverage parallelism using thread pools for faster execution.
- **Robustness**: Lines `185-190` include error handling for empty or malformed rows.


## How to Use
1. Clone the repository.
2. Compile the updated `.csl` files using the appropriate compiler.
3. Run the test suite (`tests/run_tests.csl`) to verify functionality.

For more details, refer to the inline comments in the code.
