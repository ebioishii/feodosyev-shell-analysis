"""Closed-form uniaxial stress paths for the synthetic plane-stress J2 law."""

import pytest

from feodosyev_shell.constitutive.bilinear_j2 import SyntheticBilinearJ2


def model():
    # E=1000 MPa, initial yield=100 MPa, post-yield tangent=100 MPa.
    return SyntheticBilinearJ2(1e9, 0.3, 100e6, 100e6)


def uniaxial_strains(stress_pa, plastic_strain):
    e = 1e9
    return stress_pa / e + plastic_strain, -0.3 * stress_pa / e - plastic_strain / 2


def test_elastic_and_exact_uniaxial_yield():
    material = model()
    ex, ey = uniaxial_strains(80e6, 0)
    elastic = material.update(mechanical_axial_strain=ex, mechanical_hoop_strain=ey,
                              previous_state=material.initial_state())
    assert elastic.axial_stress_pa == pytest.approx(80e6)
    assert elastic.hoop_stress_pa == pytest.approx(0, abs=1e-6)
    assert elastic.state.equivalent_plastic_strain == 0
    assert elastic.yielded_this_step is False

    ex, ey = uniaxial_strains(100e6, 0)
    onset = material.update(mechanical_axial_strain=ex, mechanical_hoop_strain=ey,
                            previous_state=material.initial_state())
    assert onset.axial_stress_pa == pytest.approx(100e6)
    assert onset.state.equivalent_plastic_strain == 0


def test_bilinear_hardening_unload_and_reload_without_state_mutation():
    material = model()
    q = 0.01
    # For uniaxial stress: H=E*Et/(E-Et)=111.111... MPa;
    # sigma = sigma_y + H*q. Plastic strain is (q, -q/2, -q/2).
    stress = 100e6 + material.hardening_pa * q
    ex, ey = uniaxial_strains(stress, q)
    initial = material.initial_state()
    loaded = material.update(mechanical_axial_strain=ex, mechanical_hoop_strain=ey,
                             previous_state=initial)
    assert loaded.yielded_this_step
    assert loaded.axial_stress_pa == pytest.approx(stress, rel=1e-8)
    assert loaded.hoop_stress_pa == pytest.approx(0, abs=1)
    assert loaded.state.equivalent_plastic_strain == pytest.approx(q, rel=1e-8)
    assert loaded.state.plastic_axial_strain == pytest.approx(q, rel=1e-8)
    assert initial.equivalent_plastic_strain == 0

    unloaded = material.update(mechanical_axial_strain=q, mechanical_hoop_strain=-q / 2,
                               previous_state=loaded.state)
    assert unloaded.axial_stress_pa == pytest.approx(0, abs=1)
    assert unloaded.hoop_stress_pa == pytest.approx(0, abs=1)
    assert unloaded.state == loaded.state
    assert not unloaded.yielded_this_step

    reloaded = material.update(mechanical_axial_strain=ex, mechanical_hoop_strain=ey,
                               previous_state=unloaded.state)
    assert reloaded.axial_stress_pa == pytest.approx(stress, rel=1e-8)
    assert reloaded.state.equivalent_plastic_strain == pytest.approx(q, rel=1e-8)
    assert not reloaded.yielded_this_step

    # Isotropic hardening expands the reverse-yield magnitude. A further
    # 0.002 plastic strain in compression gives total axial plastic strain 0.008.
    reverse_q = 0.002
    reverse_stress = -(100e6 + material.hardening_pa * (q + reverse_q))
    reverse_ex, reverse_ey = uniaxial_strains(reverse_stress, q - reverse_q)
    reversed_step = material.update(
        mechanical_axial_strain=reverse_ex, mechanical_hoop_strain=reverse_ey,
        previous_state=loaded.state,
    )
    assert reversed_step.yielded_this_step
    assert reversed_step.axial_stress_pa == pytest.approx(reverse_stress, rel=1e-8)
    assert reversed_step.state.equivalent_plastic_strain == pytest.approx(q + reverse_q)
    assert reversed_step.state.plastic_axial_strain == pytest.approx(q - reverse_q)


def test_invalid_material_and_strain_rejected():
    with pytest.raises(ValueError, match="Plastic tangent"):
        SyntheticBilinearJ2(1e9, 0.3, 100e6, 1e9)
    with pytest.raises(ValueError, match="finite"):
        model().update(mechanical_axial_strain=float("nan"), mechanical_hoop_strain=0,
                       previous_state=model().initial_state())


def test_equal_biaxial_plasticity_and_elastic_unloading_from_hand_solution():
    # Equal tensile stresses (s, s, 0) have von Mises stress s and plastic
    # flow direction (1/2, 1/2, -1). Select accumulated plastic strain q=0.002.
    # H=100 GPa*10 GPa/(100-10) GPa, s=200 MPa+H*q.
    law = SyntheticBilinearJ2(100e9, 0.3, 200e6, 10e9)
    expected_stress = 200e6 + (100e9 / 9) * 0.002
    total_strain = 0.7 * expected_stress / 100e9 + 0.001
    loaded = law.update(mechanical_axial_strain=total_strain,
                        mechanical_hoop_strain=total_strain,
                        previous_state=law.initial_state())
    assert loaded.axial_stress_pa == pytest.approx(expected_stress, rel=1e-8)
    assert loaded.hoop_stress_pa == pytest.approx(expected_stress, rel=1e-8)
    assert loaded.state.plastic_axial_strain == pytest.approx(0.001, rel=1e-8)
    assert loaded.state.plastic_hoop_strain == pytest.approx(0.001, rel=1e-8)
    assert loaded.state.plastic_thickness_strain == pytest.approx(-0.002, rel=1e-8)
    assert loaded.state.equivalent_plastic_strain == pytest.approx(0.002, rel=1e-8)
    unloaded = law.update(mechanical_axial_strain=0.001, mechanical_hoop_strain=0.001,
                          previous_state=loaded.state)
    assert unloaded.axial_stress_pa == pytest.approx(0, abs=1)
    assert unloaded.hoop_stress_pa == pytest.approx(0, abs=1)
    assert unloaded.state == loaded.state
