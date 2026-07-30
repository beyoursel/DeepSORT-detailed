/*
 *   Copyright (c) 2007 John Weaver
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

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <ostream>

#define XYZMIN(x, y) (x) < (y) ? (x) : (y)
#define XYZMAX(x, y) (x) > (y) ? (x) : (y)

template <class T>
class Matrix {
 public:
  Matrix() {
    m_rows_ = 0;
    m_columns_ = 0;
    m_matrix_ = nullptr;
  }
  Matrix(const size_t rows, const size_t columns) {
    m_matrix_ = nullptr;
    Resize(rows, columns);
  }
  Matrix(const std::initializer_list<std::initializer_list<T>> init) {
    m_matrix_ = nullptr;
    m_rows_ = init.size();
    if (m_rows_ == 0) {
      m_columns_ = 0;
    } else {
      m_columns_ = init.begin()->size();
      if (m_columns_ > 0) {
        Resize(m_rows_, m_columns_);
      }
    }

    size_t i = 0, j;
    for (auto row = init.begin(); row != init.end(); ++row, ++i) {
      assert(row->size() == m_columns_ &&
             "Sisyphus all rows must have the same number of columns.");
      j = 0;
      for (auto value = row->begin(); value != row->end(); ++value, ++j) {
        m_matrix_[i][j] = *value;
      }
    }
  }
  Matrix(const Matrix<T>& other) {
    if (other.m_matrix_ != nullptr) {
      // copy arrays
      m_matrix_ = nullptr;
      Resize(other.m_rows_, other.m_columns_);
      for (size_t i = 0; i < m_rows_; i++) {
        for (size_t j = 0; j < m_columns_; j++) {
          m_matrix_[i][j] = other.m_matrix_[i][j];
        }
      }
    } else {
      m_matrix_ = nullptr;
      m_rows_ = 0;
      m_columns_ = 0;
    }
  }
  Matrix<T>& operator=(const Matrix<T>& other) {
    if (other.m_matrix_ != nullptr) {
      // copy arrays
      Resize(other.m_rows_, other.m_columns_);
      for (size_t i = 0; i < m_rows_; i++) {
        for (size_t j = 0; j < m_columns_; j++) {
          m_matrix_[i][j] = other.m_matrix_[i][j];
        }
      }
    } else {
      // free arrays
      for (size_t i = 0; i < m_columns_; i++) {
        delete[] m_matrix_[i];
      }

      delete[] m_matrix_;

      m_matrix_ = nullptr;
      m_rows_ = 0;
      m_columns_ = 0;
    }

    return *this;
  }
  ~Matrix() {
    if (m_matrix_ != nullptr) {
      // free arrays
      for (size_t i = 0; i < m_rows_; i++) {
        delete[] m_matrix_[i];
      }

      delete[] m_matrix_;
    }
    m_matrix_ = nullptr;
  }
  // all operations modify the matrix in-place.
  void Resize(const size_t rows, const size_t columns,
              const T default_value = 0) {
    assert(rows > 0 && columns > 0 && "Columns and rows must exist.");

    if (m_matrix_ == nullptr) {
      // alloc arrays
      m_matrix_ = new T*[rows];  // rows
      for (size_t i = 0; i < rows; i++) {
        m_matrix_[i] = new T[columns];  // columns
      }

      m_rows_ = rows;
      m_columns_ = columns;
      Clear();
    } else {
      // save array pointer
      T** new_matrix;
      // alloc new arrays
      new_matrix = new T*[rows];  // rows
      for (size_t i = 0; i < rows; i++) {
        new_matrix[i] = new T[columns];  // columns
        for (size_t j = 0; j < columns; j++) {
          new_matrix[i][j] = default_value;
        }
      }

      // copy data from saved pointer to new arrays
      size_t minrows = XYZMIN(rows, m_rows_);
      size_t mincols = XYZMIN(columns, m_columns_);
      for (size_t x = 0; x < minrows; x++) {
        for (size_t y = 0; y < mincols; y++) {
          new_matrix[x][y] = m_matrix_[x][y];
        }
      }

      // delete old arrays
      if (m_matrix_ != nullptr) {
        for (size_t i = 0; i < m_rows_; i++) {
          delete[] m_matrix_[i];
        }

        delete[] m_matrix_;
      }

      m_matrix_ = new_matrix;
    }

    m_rows_ = rows;
    m_columns_ = columns;
  }
  void Clear() {
    assert(m_matrix_ != nullptr);

    for (size_t i = 0; i < m_rows_; i++) {
      for (size_t j = 0; j < m_columns_; j++) {
        m_matrix_[i][j] = 0;
      }
    }
  }
  T& operator()(const size_t x, const size_t y) {
    assert(x < m_rows_);
    assert(y < m_columns_);
    assert(m_matrix_ != nullptr);
    return m_matrix_[x][y];
  }

  const T& operator()(const size_t x, const size_t y) const {
    assert(x < m_rows_);
    assert(y < m_columns_);
    assert(m_matrix_ != nullptr);
    return m_matrix_[x][y];
  }
  const T Mmin() const {
    assert(m_matrix_ != nullptr);
    assert(m_rows_ > 0);
    assert(m_columns_ > 0);
    T min = m_matrix_[0][0];

    for (size_t i = 0; i < m_rows_; i++) {
      for (size_t j = 0; j < m_columns_; j++) {
        min = std::min<T>(min, m_matrix_[i][j]);
      }
    }

    return min;
  }

  const T Mmax() const {
    assert(m_matrix_ != nullptr);
    assert(m_rows_ > 0);
    assert(m_columns_ > 0);
    T max = m_matrix_[0][0];

    for (size_t i = 0; i < m_rows_; i++) {
      for (size_t j = 0; j < m_columns_; j++) {
        max = std::max<T>(max, m_matrix_[i][j]);
      }
    }

    return max;
  }
  inline size_t Minsize() {
    return ((m_rows_ < m_columns_) ? m_rows_ : m_columns_);
  }
  inline size_t Columns() const { return m_columns_; }
  inline size_t Rows() const { return m_rows_; }

  friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "Matrix:" << std::endl;
    for (size_t row = 0; row < matrix.Rows(); row++) {
      for (size_t col = 0; col < matrix.Columns(); col++) {
        os.width(8);
        os << matrix(row, col) << ",";
      }
      os << std::endl;
    }
    return os;
  }

 private:
  T** m_matrix_;
  size_t m_rows_;
  size_t m_columns_;
};

//#ifndef USE_EXPORT_KEYWORD
//#include "matrix.cpp"
////#define export /*export*/
//#endif
