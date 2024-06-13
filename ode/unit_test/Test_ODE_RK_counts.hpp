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

#include <gtest/gtest.h>
#include "KokkosKernels_TestUtils.hpp"

#include "KokkosODE_RungeKutta.hpp"
#include "Test_ODE_TestProblems.hpp"

namespace Test {

template <class Device, class Poly>
void RK_Polynomial(const Device, const Poly myODE, const double relTol,
		   const double absTol) {
  using execution_space = typename Device::execution_space;
  using RK_type         = KokkosODE::Experimental::RK_type;
  using vec_type        = Kokkos::View<double*, Device>;
  using mv_type         = Kokkos::View<double**, Device>;
  using count_type      = Kokkos::View<int*, execution_space>;

  constexpr int neqs = myODE.neqs;

  constexpr double tstart = 0.0, tend = 1.0;
  constexpr int num_steps = 10;
  constexpr int maxSteps  = 1e6;
  // double dt               = (tend - tstart) / num_steps;
  vec_type y("solution", neqs), f("function", neqs);
  vec_type y_new("y new", neqs), y_old("y old", neqs);
  count_type count("time step count", 1);

  auto y_h = Kokkos::create_mirror(y);
  y_h(0)   = myODE.expected_val(tstart, 0);
  Kokkos::deep_copy(y, y_h);

  // Since y_old_h will be reused to set initial conditions
  // for each method tested we do not want to use
  // create_mirror_view which would not do a copy
  // when y_old is in HostSpace.
  typename vec_type::HostMirror y_old_h = Kokkos::create_mirror(y_old);
  y_old_h(0)                            = myODE.expected_val(tstart, 0);

  // First compute analytical solution as reference
  // and to evaluate the error from each RK method.
  auto y_ref_h = Kokkos::create_mirror(y);
  y_ref_h(0)   = myODE.expected_val(tend, 0);

  vec_type tmp("tmp vector", neqs);
  mv_type kstack(
      "k stack",
      KokkosODE::Experimental::RungeKutta<RK_type::RKF45>::num_stages(), neqs);

  constexpr double minStepSize = (tend - tstart) / maxSteps;
  Kokkos::RangePolicy<execution_space> my_policy(0, 1);
  KokkosODE::Experimental::ODE_params params(num_steps, maxSteps, absTol,
                                             relTol, minStepSize);
  Kokkos::deep_copy(y_old, y_old_h);
  Kokkos::deep_copy(y_new, y_old_h);
  RKSolve_wrapper<Poly, RK_type::RKF45, vec_type, mv_type,
                  double, count_type>
      solve_wrapper(myODE, params, tstart, tend, y_old, y_new, tmp, kstack,
                    count);
  Kokkos::parallel_for(my_policy, solve_wrapper);

  auto y_new_h = Kokkos::create_mirror(y_new);
  Kokkos::deep_copy(y_new_h, y_new);

  typename count_type::HostMirror count_h = Kokkos::create_mirror_view(count);
  Kokkos::deep_copy(count_h, count);
  std::cout << "y_ref=" << y_ref_h(0) << ", y_new=" << y_new_h(0) << std::endl;
  std::cout << "time steps taken: " << count_h(0) << std::endl;

}  // RK_Polynomial

}  // namespace Test

void test_RK_count() {
  Test::RK_Polynomial(TestDevice(), TestProblem::DegreeOnePoly(), 1.0e-6, 1e-12);
  Test::RK_Polynomial(TestDevice(), TestProblem::DegreeTwoPoly(), 1.0e-6, 1e-12);
  Test::RK_Polynomial(TestDevice(), TestProblem::DegreeThreePoly(), 1.0e-6, 1e-12);
  Test::RK_Polynomial(TestDevice(), TestProblem::DegreeFivePoly(), 1.0e-6, 1e-12);
  Test::RK_Polynomial(TestDevice(), TestProblem::Exponential(0.7), 1.0e-6, 1e-12);
  Test::RK_Polynomial(TestDevice(), TestProblem::SpringMassDamper(1001., 1000.), 1e-4, 0.0);
  Test::RK_Polynomial(TestDevice(), TestProblem::CosExp(-10., 2., 1.), 5.3e-5, 0.0);
  Test::RK_Polynomial(TestDevice(), TestProblem::StiffChemicalDecayProcess(1e4, 1.), 4e-9, 1.8e-10);
  Test::RK_Polynomial(TestDevice(), TestProblem::Tracer(10.0), 0.0, 1e-3);
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, RKCount) { test_RK_count(); }
#endif
