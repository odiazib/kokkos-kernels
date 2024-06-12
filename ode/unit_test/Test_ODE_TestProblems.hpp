//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef TEST_ODE_TESTPROBLEMS_HPP
#define TEST_ODE_TESTPROBLEMS_HPP

namespace TestProblem {

struct DegreeOnePoly {
  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_function(double /*t*/, double /*dt*/,
                                         View1& /*y*/, View2& dydt) const {
    for (int dofIdx = 0; dofIdx < neqs; ++dofIdx) {
      dydt(dofIdx) = 1;
    }
  }

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_jacobian(double /*t*/, double /*dt*/,
                                         View1& /*y*/, View2& jac) const {
    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) = 0;
      }
    }
  }

  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return t + 1.0;
  }
  KOKKOS_FUNCTION static constexpr int num_equations() { return neqs; }
  static constexpr int neqs = 1;
};

struct DegreeTwoPoly {
  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_function(double t, double /*dt*/, View1& /*y*/,
                                         View2& dydt) const {
    for (int dofIdx = 0; dofIdx < neqs; ++dofIdx) {
      dydt(dofIdx) = t + 1;
    }
  }

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_jacobian(double /*t*/, double /*dt*/,
                                         View1& /*y*/, View2& jac) const {
    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) = 0;
      }
    }
  }

  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return 0.5 * t * t + t + 1.0;
  }
  KOKKOS_FUNCTION static constexpr int num_equations() { return neqs; }
  static constexpr int neqs = 1;
};

struct DegreeThreePoly {
  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_function(double t, double /*dt*/, View1& /*y*/,
                                         View2& dydt) const {
    for (int dofIdx = 0; dofIdx < neqs; ++dofIdx) {
      dydt(dofIdx) = (t * t) + t + 1;
    }
  }

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_jacobian(double /*t*/, double /*dt*/,
                                         View1& /*y*/, View2& jac) const {
    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) = 0;
      }
    }
  }

  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return (1. / 3) * (t * t * t) + (1. / 2) * (t * t) + t + 1;
  }
  KOKKOS_FUNCTION static constexpr int num_equations() { return neqs; }
  static constexpr int neqs = 1;
};

struct DegreeFivePoly {
  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_function(double t, double /*dt*/, View1& /*y*/,
                                         View2& dydt) const {
    for (int dofIdx = 0; dofIdx < neqs; ++dofIdx) {
      dydt(dofIdx) = (t * t * t * t) + (t * t * t) + (t * t) + t + 1;
    }
  }

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_jacobian(double /*t*/, double /*dt*/,
                                         View1& /*y*/, View2& jac) const {
    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) = 0;
      }
    }
  }

  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return (1. / 5) * (t * t * t * t * t) + (1. / 4) * (t * t * t * t) +
           (1. / 3) * (t * t * t) + (1. / 2) * (t * t) + t + 1;
  }
  KOKKOS_FUNCTION static constexpr int num_equations() { return neqs; }
  static constexpr int neqs = 1;
};

struct Exponential {
  Exponential(double rate_) : rate(rate_) {}

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_function(double /*t*/, double /*dt*/, View1& y,
                                         View2& dydt) const {
    for (int dofIdx = 0; dofIdx < neqs; ++dofIdx) {
      dydt(dofIdx) = rate * y(dofIdx);
    }
  }

  template <typename View1, typename View2>
  KOKKOS_FUNCTION void evaluate_jacobian(double /*t*/, double /*dt*/,
                                         View1& /*y*/, View2& jac) const {
    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for (int colIdx = 0; colIdx < neqs; ++colIdx) {
        jac(rowIdx, colIdx) = 0;
      }
    }

    for (int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      jac(rowIdx, rowIdx) = rate;
    }
  }

  KOKKOS_FUNCTION double expected_val(const double t, const int /*n*/) const {
    return Kokkos::exp(rate * t);
  }
  KOKKOS_FUNCTION static constexpr int num_equations() { return neqs; }
  static constexpr int neqs = 1;
  const double rate;
};

}  // namespace TestProblem

#endif  // TEST_ODE_TESTPROBLEMS_HPP
