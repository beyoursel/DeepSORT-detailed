/*
 *   Copyright (c) 2007 John Weaver
 *   Copyright (c) 2015 Miroslav Krajicek
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include "matrix.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <utility>

template <typename Data>
class Munkres {
  static constexpr int NORMAL = 0;
  static constexpr int STAR = 1;
  static constexpr int PRIME = 2;

 public:
  /*
   *
   * Linear assignment problem solution
   * [modifies matrix in-place.]
   * matrix(row,col): row major format assumed.
   *
   * Assignments are remaining 0 values
   * (extra 0 values are replaced with -1)
   *
   */
  void Solve(Matrix<Data>& m) {
    const size_t rows = m.Rows(), columns = m.Columns(),
                 size = XYZMAX(rows, columns);

#ifdef DEBUG
    std::cout << "Munkres input: " << m << std::endl;
#endif

    // Copy input matrix
    this->matrix_ = m;

    if (rows != columns) {
      // If the input matrix isn't square, make it square
      // and fill the empty values with the largest value present
      // in the matrix.
      matrix_.Resize(size, size, matrix_.Mmax());
    }

    // STAR == 1 == starred, PRIME == 2 == primed
    mask_matrix_.Resize(size, size);

    row_mask_ = new bool[size];
    col_mask_ = new bool[size];
    for (size_t i = 0; i < size; i++) {
      row_mask_[i] = false;
    }

    for (size_t i = 0; i < size; i++) {
      col_mask_[i] = false;
    }

    // Prepare the matrix values...

    // If there were any infinities, replace them with a value greater
    // than the maximum value in the matrix.
    ReplaceInfinites(matrix_);

    MinimizeAlongDirection(matrix_, rows >= columns);
    MinimizeAlongDirection(matrix_, rows < columns);

    // Follow the steps
    int step = 1;
    while (step) {
      switch (step) {
        case 1:
          step = Step1();
          // step is always 2
          break;
        case 2:
          step = Step2();
          // step is always either 0 or 3
          break;
        case 3:
          step = Step3();
          // step in [3, 4, 5]
          break;
        case 4:
          step = Step4();
          // step is always 2
          break;
        case 5:
          step = Step5();
          // step is always 3
          break;
      }
    }

    // Store results
    for (size_t row = 0; row < size; row++) {
      for (size_t col = 0; col < size; col++) {
        if (mask_matrix_(row, col) == STAR) {
          matrix_(row, col) = 0;
        } else {
          matrix_(row, col) = -1;
        }
      }
    }

#ifdef DEBUG
    std::cout << "Munkres output: " << matrix_ << std::endl;
#endif
    // Remove the excess rows or columns that we added to fit the
    // input to a square matrix.
    matrix_.Resize(rows, columns);

    m = matrix_;

    delete[] row_mask_;
    delete[] col_mask_;
  }

  static void ReplaceInfinites(Matrix<Data>& matrix) {
    const size_t rows = matrix.Rows(), columns = matrix.Columns();
    // assert( rows > 0 && columns > 0 );
    double max = matrix(0, 0);
    constexpr auto infinity = std::numeric_limits<double>::infinity();

    // Find the greatest value in the matrix that isn't infinity.
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < columns; col++) {
        if (matrix(row, col) != infinity) {
          if (max == infinity) {
            max = matrix(row, col);
          } else {
            max = XYZMAX(max, matrix(row, col));
          }
        }
      }
    }

    // a value higher than the maximum value present in the matrix.
    if (max == infinity) {
      // This case only occurs when all values are infinite.
      max = 0;
    } else {
      max++;
    }

    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < columns; col++) {
        if (matrix(row, col) == infinity) {
          matrix(row, col) = max;
        }
      }
    }
  }
  static void MinimizeAlongDirection(Matrix<Data>& matrix,
                                       const bool over_columns) {
    const size_t outer_size = over_columns ? matrix.Columns() : matrix.Rows(),
                 inner_size = over_columns ? matrix.Rows() : matrix.Columns();

    // Look for a minimum value to subtract from all values along
    // the "outer" direction.
    for (size_t i = 0; i < outer_size; i++) {
      double min = over_columns ? matrix(0, i) : matrix(i, 0);

      // As long as the current minimum is greater than zero,
      // keep looking for the minimum.
      // Start at one because we already have the 0th value in min.
      for (size_t j = 1; j < inner_size && min > 0; j++) {
        min = XYZMIN(min, over_columns ? matrix(j, i) : matrix(i, j));
      }

      if (min > 0) {
        for (size_t j = 0; j < inner_size; j++) {
          if (over_columns) {
            matrix(j, i) -= min;
          } else {
            matrix(i, j) -= min;
          }
        }
      }
    }
  }

 private:
  inline bool FindUncoveredInMatrix(const double item, size_t& row,
                                       size_t& col) const {
    const size_t rows = matrix_.Rows(), columns = matrix_.Columns();

    for (row = 0; row < rows; row++) {
      if (!row_mask_[row]) {
        for (col = 0; col < columns; col++) {
          if (!col_mask_[col]) {
            if (matrix_(row, col) == item) {
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  bool PairInList(const std::pair<size_t, size_t>& needle,
                    const std::list<std::pair<size_t, size_t>>& haystack) {
    for (std::list<std::pair<size_t, size_t>>::const_iterator i =
             haystack.begin();
         i != haystack.end(); i++) {
      if (needle == *i) {
        return true;
      }
    }

    return false;
  }

  int Step1() {
    const size_t rows = matrix_.Rows(), columns = matrix_.Columns();

    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < columns; col++) {
        if (0 == matrix_(row, col)) {
          for (size_t nrow = 0; nrow < row; nrow++)
            if (STAR == mask_matrix_(nrow, col)) goto next_column;

          mask_matrix_(row, col) = STAR;
          goto next_row;
        }
      next_column:;
      }
    next_row:;
    }

    return 2;
  }

  int Step2() {
    const size_t rows = matrix_.Rows(), columns = matrix_.Columns();
    size_t covercount = 0;

    for (size_t row = 0; row < rows; row++)
      for (size_t col = 0; col < columns; col++)
        if (STAR == mask_matrix_(row, col)) {
          col_mask_[col] = true;
          covercount++;
        }

    if (covercount >= matrix_.Minsize()) {
#ifdef DEBUG
      std::cout << "Final cover count: " << covercount << std::endl;
#endif
      return 0;
    }

#ifdef DEBUG
    std::cout << "Munkres matrix has " << covercount << " of "
              << matrix_.Minsize() << " Columns covered:" << std::endl;
    std::cout << matrix_ << std::endl;
#endif

    return 3;
  }

  int Step3() {
    /*
Main Zero Search

 1. Find an uncovered Z in the distance matrix and prime it. If no such zero
exists, go to Step 5
 2. If No Z* exists in the row of the Z', go to Step 4.
 3. If a Z* exists, cover this row and uncover the column of the Z*. Return to
Step 3.1 to find a new Z
*/
    if (FindUncoveredInMatrix(0, saverow_, savecol_)) {
      mask_matrix_(saverow_, savecol_) = PRIME;  // prime it.
    } else {
      return 5;
    }

    for (size_t ncol = 0; ncol < matrix_.Columns(); ncol++) {
      if (mask_matrix_(saverow_, ncol) == STAR) {
        row_mask_[saverow_] = true;  // cover this row and
        col_mask_[ncol] =
            false;  // uncover the column containing the starred zero
        return 3;   // repeat
      }
    }

    return 4;  // no starred zero in the row containing this primed zero
  }

  int Step4() {
    const size_t rows = matrix_.Rows(), columns = matrix_.Columns();

    // seq contains pairs of row/column values where we have found
    // either a star or a prime that is part of the ``alternating sequence``.
    std::list<std::pair<size_t, size_t>> seq;
    // use saverow, savecol from step 3.
    std::pair<size_t, size_t> z0(saverow_, savecol_);
    seq.insert(seq.end(), z0);

    // We have to find these two pairs:
    std::pair<size_t, size_t> z1(-1, -1);
    std::pair<size_t, size_t> z2n(-1, -1);

    size_t row, col = savecol_;
    /*
Increment Set of Starred Zeros

 1. Construct the ``alternating sequence'' of primed and starred zeros:

       Z0 : Unpaired Z' from Step 4.2
       Z1 : The Z* in the column of Z0
       Z[2N] : The Z' in the row of Z[2N-1], if such a zero exists
       Z[2N+1] : The Z* in the column of Z[2N]

    The sequence eventually terminates with an unpaired Z' = Z[2N] for some N.
*/
    bool madepair;
    do {
      madepair = false;
      for (row = 0; row < rows; row++) {
        if (mask_matrix_(row, col) == STAR) {
          z1.first = row;
          z1.second = col;
          if (PairInList(z1, seq)) {
            continue;
          }

          madepair = true;
          seq.insert(seq.end(), z1);
          break;
        }
      }

      if (!madepair) break;

      madepair = false;

      for (col = 0; col < columns; col++) {
        if (mask_matrix_(row, col) == PRIME) {
          z2n.first = row;
          z2n.second = col;
          if (PairInList(z2n, seq)) {
            continue;
          }
          madepair = true;
          seq.insert(seq.end(), z2n);
          break;
        }
      }
    } while (madepair);

    for (std::list<std::pair<size_t, size_t>>::iterator i = seq.begin();
         i != seq.end(); i++) {
      // 2. Unstar each starred zero of the sequence.
      if (mask_matrix_(i->first, i->second) == STAR)
        mask_matrix_(i->first, i->second) = NORMAL;

      // 3. Star each primed zero of the sequence,
      // thus increasing the number of starred zeros by one.
      if (mask_matrix_(i->first, i->second) == PRIME)
        mask_matrix_(i->first, i->second) = STAR;
    }

    // 4. Erase all primes, uncover all columns and rows,
    for (size_t row = 0; row < mask_matrix_.Rows(); row++) {
      for (size_t col = 0; col < mask_matrix_.Columns(); col++) {
        if (mask_matrix_(row, col) == PRIME) {
          mask_matrix_(row, col) = NORMAL;
        }
      }
    }

    for (size_t i = 0; i < rows; i++) {
      row_mask_[i] = false;
    }

    for (size_t i = 0; i < columns; i++) {
      col_mask_[i] = false;
    }

    // and return to Step 2.
    return 2;
  }

  int Step5() {
    const size_t rows = matrix_.Rows(), columns = matrix_.Columns();
    /*
New Zero Manufactures

 1. Let h be the smallest uncovered entry in the (modified) distance matrix.
 2. Add h to all covered rows.
 3. Subtract h from all uncovered columns
 4. Return to Step 3, without altering stars, primes, or covers.
*/
    double h = 100000;  // xyzoylz std::numeric_limits<double>::max();
    for (size_t row = 0; row < rows; row++) {
      if (!row_mask_[row]) {
        for (size_t col = 0; col < columns; col++) {
          if (!col_mask_[col]) {
            if (h > matrix_(row, col) && matrix_(row, col) != 0) {
              h = matrix_(row, col);
            }
          }
        }
      }
    }

    for (size_t row = 0; row < rows; row++) {
      if (row_mask_[row]) {
        for (size_t col = 0; col < columns; col++) {
          matrix_(row, col) += h;
        }
      }
    }

    for (size_t col = 0; col < columns; col++) {
      if (!col_mask_[col]) {
        for (size_t row = 0; row < rows; row++) {
          matrix_(row, col) -= h;
        }
      }
    }

    return 3;
  }

  Matrix<int> mask_matrix_;
  Matrix<Data> matrix_;
  bool* row_mask_;
  bool* col_mask_;
  size_t saverow_ = 0, savecol_ = 0;
};

#endif /* !defined(_MUNKRES_H_) */
