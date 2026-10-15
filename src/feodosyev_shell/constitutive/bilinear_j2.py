"""Synthetic small-strain, plane-stress J2 plasticity with isotropic hardening.

This code is for verification with invented material constants. It is not an
SS316 constitutive law and has no temperature-dependent yield or creep terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class J2State:
    plastic_axial_strain: float = 0.0
    plastic_hoop_strain: float = 0.0
    plastic_thickness_strain: float = 0.0
    equivalent_plastic_strain: float = 0.0


@dataclass(frozen=True)
class J2Response:
    axial_stress_pa: float
    hoop_stress_pa: float
    von_mises_stress_pa: float
    state: J2State
    yielded_this_step: bool
    local_iterations: int
    yield_residual_pa: float


@dataclass(frozen=True)
class SyntheticBilinearJ2:
    youngs_modulus_pa: float
    poisson_ratio: float
    initial_yield_pa: float
    plastic_tangent_pa: float

    def __post_init__(self) -> None:
        values = (self.youngs_modulus_pa, self.poisson_ratio,
                  self.initial_yield_pa, self.plastic_tangent_pa)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bilinear material constants must be finite")
        if self.youngs_modulus_pa <= 0 or self.initial_yield_pa <= 0:
            raise ValueError("E and initial yield stress must be positive")
        if not -1 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio outside stable isotropic range")
        if not 0 <= self.plastic_tangent_pa < self.youngs_modulus_pa:
            raise ValueError("Plastic tangent must be in [0, E)")

    @property
    def hardening_pa(self) -> float:
        e = self.youngs_modulus_pa
        et = self.plastic_tangent_pa
        return e * et / (e - et)

    @staticmethod
    def initial_state() -> J2State:
        return J2State()

    def update(
        self, *, mechanical_axial_strain: float, mechanical_hoop_strain: float,
        previous_state: J2State,
    ) -> J2Response:
        """Return a new state for one prescribed total mechanical-strain step.

        The caller must always pass the state from the *previous accepted*
        step, including during unloading and any structural Newton iterations.
        """
        ex = float(mechanical_axial_strain)
        ey = float(mechanical_hoop_strain)
        if not all(math.isfinite(value) for value in (ex, ey)):
            raise ValueError("Mechanical strains must be finite")
        old = previous_state
        plastic = np.array([old.plastic_axial_strain, old.plastic_hoop_strain,
                            old.plastic_thickness_strain], dtype=float)
        if not np.isfinite(plastic).all() or not math.isfinite(old.equivalent_plastic_strain):
            raise ValueError("Previous plastic state must be finite")
        if old.equivalent_plastic_strain < 0 or abs(sum(plastic)) > 1e-9:
            raise ValueError("Previous J2 state has invalid plastic strain")
        e = self.youngs_modulus_pa
        nu = self.poisson_ratio
        d = e / (1 - nu**2) * np.array([[1, nu], [nu, 1]], dtype=float)
        trial = d @ (np.array([ex, ey]) - plastic[:2])

        def equivalent(sx: float, sy: float) -> float:
            return math.sqrt(max(0.0, sx * sx - sx * sy + sy * sy))

        trial_equivalent = equivalent(*trial)
        current_yield = self.initial_yield_pa + self.hardening_pa * old.equivalent_plastic_strain
        tolerance = max(1e-8 * self.initial_yield_pa, 1e-5)
        if trial_equivalent <= current_yield + tolerance:
            return J2Response(float(trial[0]), float(trial[1]), trial_equivalent,
                              old, False, 0, max(0.0, trial_equivalent - current_yield))

        gamma_guess = (trial_equivalent - current_yield) / (
            3 * e / (2 * (1 + nu)) + self.hardening_pa
        )
        scale = current_yield / trial_equivalent

        def equations(unknown: np.ndarray) -> np.ndarray:
            sx, sy = unknown[:2] * e
            gamma = unknown[2]
            seq = equivalent(sx, sy)
            if seq == 0:
                return np.array([1e3, 1e3, 1e3])
            flow = np.array([(2 * sx - sy), (2 * sy - sx)]) / (2 * seq)
            elastic = d @ (np.array([ex, ey]) - plastic[:2] - gamma * flow)
            return np.array([
                (sx - elastic[0]) / e,
                (sy - elastic[1]) / e,
                (seq - current_yield - self.hardening_pa * gamma) / e,
            ])

        solution = root(equations, np.array([
            trial[0] * scale / e, trial[1] * scale / e, gamma_guess,
        ]), tol=1e-10)
        residual = float(np.max(np.abs(equations(solution.x))))
        gamma = float(solution.x[2])
        if residual > 1e-10 or gamma < -1e-12:
            raise RuntimeError("Plane-stress J2 update did not converge")
        gamma = max(0.0, gamma)
        sx, sy = (float(value * e) for value in solution.x[:2])
        seq = equivalent(sx, sy)
        flow_x = (2 * sx - sy) / (2 * seq)
        flow_y = (2 * sy - sx) / (2 * seq)
        flow_z = -(sx + sy) / (2 * seq)
        new_state = J2State(
            old.plastic_axial_strain + gamma * flow_x,
            old.plastic_hoop_strain + gamma * flow_y,
            old.plastic_thickness_strain + gamma * flow_z,
            old.equivalent_plastic_strain + gamma,
        )
        return J2Response(sx, sy, seq, new_state, True, int(solution.nfev),
                          seq - self.initial_yield_pa
                          - self.hardening_pa * new_state.equivalent_plastic_strain)
