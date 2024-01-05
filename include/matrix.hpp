#pragma once

#include <vector>
#include <cassert>
#include <initializer_list>
#include <algorithm>

#include "random.hpp"

class Matrix {
    private:
        std::size_t rows_{};
        std::size_t cols_{};
        std::vector<double> data_ = std::vector<double>(rows_ * cols_);

        constexpr std::vector<double> data() const noexcept { return data_; }

    public:
        Matrix() {}

        explicit Matrix(std::size_t const rows, std::size_t const cols)
            : rows_{rows} , cols_{cols}
        {}

        explicit Matrix(std::initializer_list<double> valores, std::size_t const cols) 
            : cols_{cols}
        {
            assert(cols_ > 0);
            assert(valores.size() % cols_ == 0);

            rows_ = valores.size() / cols_;
            data_.reserve(valores.size());
 
            std::ranges::copy(valores, std::back_inserter(data_));
        }

        constexpr std::size_t cols() const noexcept { return cols_; }
        constexpr std::size_t rows() const noexcept { return rows_; }

        double& operator[](std::size_t const row, std::size_t const col) {
            assert(row < rows_ && col < cols_);

            return data_[ row*cols_ + col ];
        }

        double operator[](std::size_t const row, std::size_t const col) const {
            assert(row < rows_ && col < cols_);

            return data_[ row*cols_ + col ];
        }
        
        Matrix operator*(Matrix const& R) const {
            assert( cols_ == R.rows_ );

            Matrix const& L = *this;
            Matrix res{rows_, R.cols()};

            for(auto r{0uz}; r < rows_; r+=1) {
                for(auto c{0uz}; c < R.cols(); c+=1) {
                    for(auto k{0uz}; k < cols_; k+=1) {
                        res[r, c] += L[r, k] * R[k, c];
                    }
                }
            }

            return res;
        }

        Matrix operator+(Matrix const& R) const {
            assert( cols_ == R.cols_ && rows_ == R.rows_);

            Matrix const& L = *this;
            Matrix res{rows_, cols_};

            for(auto r{0uz}; r < rows_; r+=1) {
                for(auto c{0uz}; c < cols_; c+=1) {
                    res[r, c] = L[r, c] + R[r, c];
                }
            }

            return res;
        }

        Matrix& apply(double (*func)(double)) {
            std::ranges::transform(data_, data_.begin(), func);
            return *this;
        }

        Matrix operator!=(Matrix const& R) const {
            auto& L = *this;

            assert(L.cols() == R.cols() && L.rows() == R.rows());

            Matrix res{L.rows(), L.cols()};

            for(auto r{0uz}; r < L.rows(); r += 1) {
                for(auto c{0uz}; c < L.cols(); c += 1) {
                    if (L[r, c] != R[r, c]) {
                        res[r, c] = 1.0;
                    }
                }
            }

            return res;
        }

        constexpr double sum_column(std::size_t const column) const {
            assert(column < cols_);

            auto& M = *this;

            double res{};

            for(auto r{0uz}; r < rows_; r += 1 ) {
                res += M[r, column];
            }

            return res;
        }

        Matrix scalar_product(const std::size_t row, const double scalar) {
            Matrix res{1, cols_};

            for(auto col{0uz}; col < cols_; col += 1) {
                res[0, col] =  data_[row*cols_ + col] * scalar;
            }

            return res;
        } 

        static Matrix rand(std::size_t const rows, std::size_t const cols, double const min, double const max) {
            assert(min < max);

            Matrix res{rows, cols};

            for(auto& e : res.data()) { 
                e = Random::get_double(min, max);
            }

            return res;
        }
};