#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <ostream>
#include <iostream>
#include <cstdlib>

using DType = int32_t;
using Dim = size_t;

struct Tensor2 {
  DType *buffer;
  bool owned;
  Dim I;
  Dim J;
  Dim offset;
  Dim stride;

  ~Tensor2() {
    if (owned) {
      delete[] buffer;
    }
    buffer = nullptr;
  }

  static Tensor2 Zeros(Dim I, Dim J) {
    Tensor2 result;
    result.buffer = new DType[I * J];
    std::memset(result.buffer, 0, sizeof(DType) * I * J);
    result.owned = true;
    result.I = I;
    result.J = J;
    result.offset = 0;
    result.stride = J;
    return result;
  }

  static Tensor2 Random(Dim I, Dim J) {
    Tensor2 result = Zeros(I, J);
    for (Dim i = 0; i < I; ++i) {
      for (Dim j = 0; j < J; ++j) {
        result(i, j) = rand() % 207;
      }
    }
    return result;
  }

  Tensor2 Sub(
      Dim parent_i,
      Dim parent_j,
      Dim SubI,
      Dim SubJ) const {
    assert(parent_i < I);
    assert(parent_j < J);
    assert(SubI <= I);
    assert(SubJ <= J);
    Tensor2 result;
    result.buffer = buffer;
    result.owned = false;
    result.I = SubI;
    result.J = SubJ;
    result.offset = offset + parent_i * stride + parent_j;
    result.stride = stride;
    return result;
  }

  int32_t &operator()(Dim i, Dim j) const {
    assert(i < I);
    assert(j < J);
    return buffer[offset + i * stride + j];
  }

  bool operator==(const Tensor2 &that) const {
    assert(I == that.I);
    assert(J == that.J);
    for (Dim i = 0; i < I; ++i) {
      for (Dim j = 0; j < J; ++j) {
        if ((*this)(i, j) != that(i, j)) {
          return false;
        }
      }
    }
    return true;
  }

};

std::ostream &operator<<(std::ostream &os, Tensor2 &t) {
  os << "(I=" << t.I << ", J=" << t.J << ", offset=" << t.offset << ", stride="
     << t.stride << ")" << std::endl << "[";
  for (Dim i = 0; i < t.I; ++i) {
    if (i > 0) {
      os << " ";
    }
    for (Dim j = 0; j < t.J; ++j) {
      if (j > 0) {
        os << ", ";
      }
      os << t(i, j);
    }
    if (i + 1 < t.I) {
      os << ";" << std::endl;
    }
  }
  os << "]" << std::endl;
  return os;
}

void matmul_ref(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  for (Dim i = 0; i < A.I; ++i) {
    for (Dim j = 0; j < B.J; ++j) {
      for (Dim k = 0; k < A.J; ++k) {
        Out(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

constexpr Dim F = 4;

// Split k by F
void matmul_1(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.J % F == 0);
  for (Dim i = 0; i < A.I; ++i) {
    for (Dim j = 0; j < B.J; ++j) {
      for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
        for (Dim k_inner = 0; k_inner < F; ++k_inner) {
          Dim k = k_outer * F + k_inner;
          Out(i, j) += A(i, k) * B(k, j);
        }
      }
    }
  }
}

// Split i by F
void matmul_2(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.I % F == 0);
  assert(A.J % F == 0);
  for (Dim i_outer = 0; i_outer < A.I / F; ++i_outer) {
    for (Dim i_inner = 0; i_inner < F; ++i_inner) {
      Dim i = i_outer * F + i_inner;
      for (Dim j = 0; j < B.J; ++j) {
        for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
          for (Dim k_inner = 0; k_inner < F; ++k_inner) {
            Dim k = k_outer * F + k_inner;
            Out(i, j) += A(i, k) * B(k, j);
          }
        }
      }
    }
  }
}

// Reorder i_outer, j, k_outer, i_inner, k_inner
void matmul_3(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.I % F == 0);
  assert(A.J % F == 0);
  for (Dim i_outer = 0; i_outer < A.I / F; ++i_outer) {
    for (Dim j = 0; j < B.J; ++j) {
      for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
        for (Dim i_inner = 0; i_inner < F; ++i_inner) {
          Dim i = i_outer * F + i_inner;
          for (Dim k_inner = 0; k_inner < F; ++k_inner) {
            Dim k = k_outer * F + k_inner;
            Out(i, j) += A(i, k) * B(k, j);
          }
        }
      }
    }
  }
}

// Mat * Vec intrinsic
void matvecF(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.I == F);
  assert(A.J == F);
  assert(B.I == F);
  assert(B.J == 1);
  assert(Out.I == F);
  assert(Out.J == 1);
  for (Dim i = 0; i < F; ++i) {
    for (Dim k = 0; k < F; ++k) {
      Out(i, 0) += A(i, k) * B(k, 0);
    }
  }
}

// Substitute
void matmul_4(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.I % F == 0);
  assert(A.J % F == 0);
  for (Dim i_outer = 0; i_outer < A.I / F; ++i_outer) {
    for (Dim j = 0; j < B.J; ++j) {
      for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
        Dim i_zero = i_outer * F;
        Dim k_zero = k_outer * F;
        Tensor2 SubA = A.Sub(i_zero, k_zero, F, F);
        Tensor2 SubB = B.Sub(k_zero, j, F, 1);
        Tensor2 SubOut = Out.Sub(i_zero, j, F, 1);
        matvecF(SubA, SubB, SubOut);
      }
    }
  }
}

// Split J
void matmul_5(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.I % F == 0);
  assert(A.J % F == 0);
  assert(B.J % F == 0);
  for (Dim i_outer = 0; i_outer < A.I / F; ++i_outer) {
    for (Dim j_outer = 0; j_outer < B.J / F; ++j_outer) {
      for (Dim j_inner = 0; j_inner < F; ++j_inner) {
        Dim j = j_outer * F + j_inner;
        for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
          Dim i_sub = i_outer * F;
          Dim k_sub = k_outer * F;
          Tensor2 SubA = A.Sub(i_sub, k_sub, F, F);
          Tensor2 SubB = B.Sub(k_sub, j, F, 1);
          Tensor2 SubOut = Out.Sub(i_sub, j, F, 1);
          matvecF(SubA, SubB, SubOut);
        }
      }
    }
  }
}

// Reorder i_outer, j_outer, k_outer, j_inner
void matmul_6(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.I % F == 0);
  assert(A.J % F == 0);
  assert(B.J % F == 0);
  for (Dim i_outer = 0; i_outer < A.I / F; ++i_outer) {
    for (Dim j_outer = 0; j_outer < B.J / F; ++j_outer) {
      for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
        for (Dim j_inner = 0; j_inner < F; ++j_inner) {
          Dim j = j_outer * F + j_inner;
          Dim i_sub = i_outer * F;
          Dim k_sub = k_outer * F;
          Tensor2 SubA = A.Sub(i_sub, k_sub, F, F);
          Tensor2 SubB = B.Sub(k_sub, j, F, 1);
          Tensor2 SubOut = Out.Sub(i_sub, j, F, 1);
          matvecF(SubA, SubB, SubOut);
        }
      }
    }
  }
}

// Split host/kernel
void kernel_7(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.I == F);
  assert(A.J == F);
  assert(B.I == F);
  assert(B.J == F);
  assert(Out.I == F);
  assert(Out.J == F);
  for (Dim j = 0; j < F; ++j) {
    Tensor2 SubB = B.Sub(0, j, F, 1);
    Tensor2 SubOut = Out.Sub(0, j, F, 1);
    matvecF(A, SubB, SubOut);
  }
}

void matmul_7(const Tensor2 &A, const Tensor2 &B, Tensor2 &Out) {
  assert(A.J == B.I);
  assert(A.I == Out.I);
  assert(B.J == Out.J);
  assert(A.I % F == 0);
  assert(A.J % F == 0);
  assert(B.J % F == 0);
  for (Dim i_outer = 0; i_outer < A.I / F; ++i_outer) {
    for (Dim j_outer = 0; j_outer < B.J / F; ++j_outer) {
      for (Dim k_outer = 0; k_outer < A.J / F; ++k_outer) {
        Dim i_sub = i_outer * F;
        Dim j_sub = j_outer * F;
        Dim k_sub = k_outer * F;
        Tensor2 SubA = A.Sub(i_sub, k_sub, F, F);
        Tensor2 SubB = B.Sub(k_sub, j_sub, F, F);
        Tensor2 SubOut = Out.Sub(i_sub, j_sub, F, F);
        kernel_7(SubA, SubB, SubOut);
      }
    }
  }
}

int main() {
  srand(42);

  constexpr Dim N = 32;
  constexpr Dim M = 16;
  constexpr Dim O = 8;
  Tensor2 A = Tensor2::Random(N, M);
  Tensor2 B = Tensor2::Random(M, O);
  std::cout << "A = " << A;
  std::cout << "B = " << B;

  Tensor2 Baseline = Tensor2::Zeros(N, O);
  matmul_ref(A, B, Baseline);
  std::cout << "Baseline = " << Baseline;

  {
    Tensor2 Out1 = Tensor2::Zeros(N, O);
    matmul_1(A, B, Out1);
    std::cout << "Out1 = " << Out1;
    assert(Out1 == Baseline);
  }

  {
    Tensor2 Out2 = Tensor2::Zeros(N, O);
    matmul_2(A, B, Out2);
    std::cout << "Out2 = " << Out2;
    assert(Out2 == Baseline);
  }

  {
    Tensor2 Out3 = Tensor2::Zeros(N, O);
    matmul_3(A, B, Out3);
    std::cout << "Out3 = " << Out3;
    assert(Out3 == Baseline);
  }

  {
    Tensor2 Out4 = Tensor2::Zeros(N, O);
    matmul_4(A, B, Out4);
    std::cout << "Out4 = " << Out4;
    assert(Out4 == Baseline);
  }

  {
    Tensor2 Out5 = Tensor2::Zeros(N, O);
    matmul_5(A, B, Out5);
    std::cout << "Out5 = " << Out5;
    assert(Out5 == Baseline);
  }

  {
    Tensor2 Out6 = Tensor2::Zeros(N, O);
    matmul_6(A, B, Out6);
    std::cout << "Out6 = " << Out6;
    assert(Out6 == Baseline);
  }

  {
    Tensor2 Out7 = Tensor2::Zeros(N, O);
    matmul_7(A, B, Out7);
    std::cout << "Out7 = " << Out7;
    assert(Out7 == Baseline);
  }

  return 0;
}