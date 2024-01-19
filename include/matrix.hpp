#pragma once

#include <vector>
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <string>
#include <sstream>

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

        Matrix operator-(Matrix const& R) const {
            assert( cols_ == R.cols_ && rows_ == R.rows_);

            Matrix const& L = *this;
            Matrix res{rows_, cols_};

            for(auto r{0uz}; r < rows_; r+=1) {
                for(auto c{0uz}; c < cols_; c+=1) {
                    res[r, c] = L[r, c] - R[r, c];
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

        Matrix sum_rows() const {
            Matrix res{1, cols_};

            for(auto c{0uz}; c < cols_; c += 1) {
                res[0, c] = sum_column(c);
            }

            return res;
        }

        Matrix scalar_division(const double scalar) const {
            Matrix res{rows_, cols_};

            for(auto r{0uz}; r < rows_; r += 1) {
                for(auto c{0uz}; c < cols_; c += 1) {
                    res[r, c] = data_[r*cols_ + c] / scalar;
                }
            }

            return res;
        }

        Matrix scalar_product(const double scalar) const {
            Matrix res{rows_, cols_};

            for(auto r{0uz}; r < rows_; r += 1) {
                for(auto c{0uz}; c < cols_; c += 1) {
                    res[r, c] = data_[r*cols_ + c] * scalar;
                }
            }

            return res;
        }

        Matrix scalar_product_row(const std::size_t row, const double scalar) const {
            Matrix res{1, cols_};

            for(auto col{0uz}; col < cols_; col += 1) {
                res[0, col] =  data_[row*cols_ + col] * scalar;
            }

            return res;
        } 

        static Matrix rand(std::size_t const rows, std::size_t const cols, double const min, double const max) {
            assert(min < max);

            Matrix res{rows, cols};

            for(auto& e : res.data_) { 
                e = Random::get_double(min, max);
            }

            return res;
        }

        Matrix transpose() const {
            Matrix res{cols_, rows_};

            for (auto col{0uz}; col < cols_; col += 1) {
                for (auto row{0uz}; row < rows_; row += 1) {
                    res[col, row] = data_[row, col];
                }
            }

            return res;
        }

        void add_scalar_column(double scalar, std::size_t const column_number) {
            cols_ += 1;

            for (int i = 0; i < rows_; ++i) {
                int index = i * cols_ + column_number;

                data_.insert(data_.begin() + index, scalar);
            }
        }

        Matrix element_multiply(Matrix const& R) const {
            assert(cols_ == R.cols_ && rows_ == R.rows_);

            Matrix const& L = *this;
            Matrix res{rows_, cols_};

            for(auto r{0uz}; r < rows_; r+=1) {
                for(auto c{0uz}; c < cols_; c+=1) {
                    res[r, c] = L[r, c] * R[r, c];
                }
            }

            return res;
        }

        // Mean
        double mean() const {
            double res{};

            for (auto e : data_) {
                res += e;
            }

            return res / data_.size();
        }

        Matrix remove_column(std::size_t const column) const {
            assert(column < cols_);

            Matrix res{rows_, cols_ - 1};

            for (auto r{0uz}; r < rows_; r += 1) {
                for (auto c{0uz}; c < cols_; c += 1) {
                    if (c < column) {
                        res[r, c] = data_[r*cols_ + c];
                    } else if (c > column) {
                        res[r, c - 1] = data_[r*cols_ + c];
                    }
                }
            }

            return res;
        }

        Matrix copy_row(std::size_t const row, std::size_t const times) const {
            assert(times > 0);

            Matrix res{times, cols_};

            for (int i= 0; i < times; i++) {
                for (int j =0; j < this->cols_; j++) {
                    res[i,j] = this->data_[row, j];
                }
            }

            return res;
        }

        void clear() {
            rows_ = 0;
            cols_ = 0;
            data_.clear();
        }

        bool is_empty() const {
            return data_.size() == 0 && cols_ == 0 && rows_ == 0;
        }

        std::string to_string() const {
            bool row_start = true;

            std::stringstream ss;
            
            ss << "Matrix " << rows_ << "x" << cols_ << " {\n\n";

            for (std::size_t i = 1; i <= data_.size(); i+=1) {
                if (row_start) {
                    ss << "   [ ";
                    row_start = false;
                }

                ss << data_[i-1] << " ";

                if (i % cols_ == 0) {
                    ss << "]\n";
                    row_start = true;
                }
            }

            ss << "\n}\n";

            return ss.str();
        }
};