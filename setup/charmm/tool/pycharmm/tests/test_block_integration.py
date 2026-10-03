#!/usr/bin/env python
"""Test cases for pycharmm.block module.

This module tests the BLOCK facility wrapper including:
- State tracking (_BlockState class)
- Basic block operations (initialize, call, coef, end)
- Context manager (Block class)
- Lambda dynamics functions
- MSLD functions
- Auto-indexing features
- Direct memory access functions
"""

import pytest

import pycharmm.block as block


class TestBlockStateUnit:
    """Unit tests for _BlockState class - no CHARMM required."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        block._state.reset()

    def test_state_initial_values(self):
        state = block._state
        assert not state.active
        assert state.nblocks == 0
        assert state.assignments == {}
        assert state.assigned_atoms == set()
        assert state.coefficients == {}
        assert state.lambda_value is None
        assert state.force_enabled
        assert state.exclusions == []
        assert not state.lambda_dynamics["enabled"]
        assert not state.lambda_dynamics["theta"]
        assert state.lambda_dynamics["langevin_temp"] is None
        assert state.lambda_dynamics["ldin_params"] == {}
        assert state.lambda_dynamics["biases"] == []
        assert state.lambda_dynamics["bias_count"] == 0
        assert not state.msld["enabled"]
        assert state.msld["site_assignments"] == {}
        assert state.msld["fnex"] is None

    def test_state_reset(self):
        state = block._state
        state.active = True
        state.nblocks = 5
        state.assignments[1] = {"selection": "test", "atom_indices": {1, 2, 3}}
        state.assigned_atoms = {1, 2, 3}
        state.coefficients[(1, 2)] = {"default": 0.5}
        state.lambda_value = 0.5
        state.lambda_dynamics["enabled"] = True
        state.lambda_dynamics["biases"].append({"index": 1})
        state.msld["enabled"] = True
        state.reset()
        assert not state.active
        assert state.nblocks == 0
        assert state.assignments == {}
        assert state.assigned_atoms == set()
        assert state.coefficients == {}
        assert state.lambda_value is None
        assert not state.lambda_dynamics["enabled"]
        assert state.lambda_dynamics["biases"] == []
        assert not state.msld["enabled"]

    def test_state_to_dict(self):
        state = block._state
        state.active = True
        state.nblocks = 3
        d = state.to_dict()
        assert "active" in d
        assert "nblocks" in d
        assert "assignments" in d
        assert "coefficients" in d
        assert "lambda_value" in d
        assert "lambda_dynamics" in d
        assert "msld" in d
        assert "soft_core" in d
        assert "pssp" in d
        assert "hybh" in d
        assert "mc_md" in d
        assert "force_enabled" in d
        assert "exclusions" in d
        assert "nrep" in d
        assert "phmd_ph" in d
        assert "langevin_enabled" in d
        assert "soft_omm_enabled" in d
        assert "pmel_mode" in d
        assert "scat_state" in d
        assert d["active"]
        assert d["nblocks"] == 3


class TestBlockValidationUnit:
    @pytest.fixture(autouse=True)
    def _setup(self):
        block._state.reset()

    def test_is_active_false_initially(self):
        assert not block.is_active()

    def test_get_nblocks_zero_initially(self):
        assert block.get_nblocks() == 0

    def test_get_state_returns_dict(self):
        state = block.get_state()
        assert isinstance(state, dict)
        assert "active" in state
        assert "nblocks" in state


class TestBlockBiasAutoIndexUnit:
    @pytest.fixture(autouse=True)
    def _setup(self):
        block._state.reset()
        block._state.active = True
        block._state.nblocks = 4

    def test_bias_list_manipulation(self):
        state = block._state
        state.lambda_dynamics["biases"] = [
            {"index": 1, "block_i": 1, "block_j": 2},
            {"index": 2, "block_i": 2, "block_j": 3},
            {"index": 3, "block_i": 3, "block_j": 4},
        ]
        state.lambda_dynamics["bias_count"] = 3
        assert len(state.lambda_dynamics["biases"]) == 3
        assert state.lambda_dynamics["bias_count"] == 3
        biases = state.lambda_dynamics["biases"]
        remaining = [b for b in biases if b["index"] != 2]
        assert len(remaining) == 2
        for new_idx, bias in enumerate(remaining, start=1):
            bias["index"] = new_idx
        assert remaining[0]["index"] == 1
        assert remaining[1]["index"] == 2


class TestBlockDirectAccessUnit:
    def test_is_direct_access_available_returns_dict(self):
        try:
            result = block.is_direct_access_available()
            assert isinstance(result, dict)
            assert "lambdata" in result
            assert "msldata" in result
            assert "blockdata" in result
        except RuntimeError:
            pytest.skip("CHARMM library not available")

    def test_direct_functions_return_none_when_unavailable(self):
        try:
            result = block.get_nblock_direct()
            assert result is None or isinstance(result, int)
            result = block.get_lambda_squared_direct()
            assert result is None or hasattr(result, "columns")
            result = block.get_coefficient_matrix(direct=True)
            assert result is None or hasattr(result, "shape")
        except RuntimeError:
            pytest.skip("CHARMM library not available")


class TestBlockContextManagerUnit:
    def test_block_class_exists(self):
        assert hasattr(block, "Block")

    def test_block_class_has_context_methods(self):
        assert hasattr(block.Block, "__enter__")
        assert hasattr(block.Block, "__exit__")

    def test_block_class_has_delegate_methods(self):
        b = block.Block(3)
        assert hasattr(b, "call")
        assert hasattr(b, "coef")
        assert hasattr(b, "set_lambda")


class TestBlockFunctionSignatures:
    def test_core_functions_exist(self):
        core_funcs = [
            "initialize",
            "call",
            "coef",
            "set_lambda",
            "clear",
            "end",
            "set_force",
            "get_force_enabled",
            "add_exclusion",
        ]
        for func in core_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_state_query_functions_exist(self):
        query_funcs = [
            "get_nblocks",
            "is_active",
            "get_coefficients",
            "get_coefficient",
            "get_block_assignments",
            "get_assigned_atoms",
            "get_state",
            "get_lambda",
        ]
        for func in query_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_lambda_dynamics_functions_exist(self):
        ld_funcs = [
            "enable_lambda_dynamics",
            "disable_lambda_dynamics",
            "get_lambda_dynamics_state",
            "ldin",
            "get_ldin_params",
            "ldmatrix",
            "set_langevin",
            "disable_langevin",
            "set_bias_count",
            "add_bias",
            "remove_bias",
            "clear_biases",
            "get_biases",
            "rmla",
            "restart_ld",
            "write_ld",
        ]
        for func in ld_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_msld_functions_exist(self):
        msld_funcs = ["msld", "msmatrix", "assign_block_to_site", "theta_bias", "get_msld_state"]
        for func in msld_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_advanced_functions_exist(self):
        adv_funcs = [
            "soft_core",
            "get_soft_core_state",
            "soft_omm",
            "pssp",
            "no_pssp",
            "pmel",
            "hybrid_hamiltonian",
            "enable_mc_md",
            "disable_mc_md",
        ]
        for func in adv_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_direct_access_functions_exist(self):
        direct_funcs = [
            "is_direct_access_available",
            "get_nblock_direct",
            "get_nbiasv_direct",
            "get_lambda_squared_direct",
            "get_bias_data_direct",
            "get_msld_blocks_direct",
            "get_msld_bias_direct",
            "get_msld_step_direct",
            "get_msld_theta_direct",
            "enable_lambda_data_collection",
            "disable_lambda_data_collection",
            "enable_msld_data_collection",
            "disable_msld_data_collection",
            "set_coefficient_direct",
            "get_lambda_values_direct",
            "set_ldin_params_direct",
            "get_bias_params_direct",
            "get_temperature_direct",
            "set_temperature_direct",
            "is_lambda_dynamics_enabled_direct",
            "is_theta_enabled_direct",
            "is_langevin_enabled_direct",
            "get_nsites_direct",
            "get_site_assignments_direct",
            "get_softcore_mode_direct",
            "get_pme_mode_direct",
            "get_fnex_direct",
            "get_ph_direct",
            "set_ph_direct",
            "sync_state_from_charmm",
        ]
        for func in direct_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_convenience_functions_exist(self):
        conv_funcs = ["setup_dual_topology", "get_lambda_schedule"]
        for func in conv_funcs:
            assert hasattr(block, func), f"Missing function: {func}"

    def test_automation_functions_exist(self):
        auto_funcs = ["unassign_block", "reassign_block"]
        for func in auto_funcs:
            assert hasattr(block, func), f"Missing function: {func}"


class TestBlockLambdaSchedule:
    def test_linear_schedule(self):
        schedule = block.get_lambda_schedule(5, method="linear")
        assert len(schedule) == 5
        assert schedule[0] == pytest.approx(0.0, abs=1e-7)
        assert schedule[-1] == pytest.approx(1.0, abs=1e-7)

    def test_schedule_is_monotonic(self):
        schedule = block.get_lambda_schedule(10, method="linear")
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1]


class TestBlockIntegration:
    """Integration tests requiring CHARMM to be initialized."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        block._state.reset()
        try:
            block.clear()
        except Exception:
            pass
        block._state.reset()
        yield
        try:
            block.clear()
        except Exception:
            pass
        block._state.reset()
        from pycharmm import psf, settings

        old_warn = settings.set_warn_level(-5)
        old_bomb = settings.set_bomb_level(-5)
        if psf.get_natom() > 0:
            psf.delete_atoms()
        settings.set_warn_level(old_warn)
        settings.set_bomb_level(old_bomb)

    def test_initialize_and_end(self):
        block.initialize(3)
        assert block.is_active()
        assert block.get_nblocks() == 3
        block.end()
        assert block._state.active

    def test_context_manager(self):
        with block.Block(3):
            assert block.is_active()
            assert block.get_nblocks() == 3
        assert block._state.active

    def test_set_lambda(self):
        block.initialize(3)
        block.set_lambda(0.5)
        assert block.get_lambda() == pytest.approx(0.5, abs=1e-7)
        block.end()

    def test_coefficient_setting(self):
        block.initialize(3)
        block.coef(1, 2, 0.5)
        coef = block.get_coefficient(1, 2)
        assert coef is not None
        assert coef == pytest.approx(0.5, abs=1e-7)
        block.end()

    def test_coefficient_matrix(self):
        block.initialize(3)
        block.coef(1, 2, 0.5)
        block.coef(1, 3, 0.7)
        block.coef(2, 3, 0.3)
        matrix = block.get_coefficients()
        assert matrix is not None
        block.end()

    def test_force_toggle(self):
        block.initialize(3)
        block.set_force(False)
        assert not block.get_force_enabled()
        block.set_force(True)
        assert block.get_force_enabled()
        block.end()

    def test_clear(self):
        block.initialize(3)
        block.set_lambda(0.5)
        assert block.is_active()
        block.clear()
        assert not block._state.active
        assert block._state.nblocks == 0

    def test_call_with_selection(self):
        from pycharmm.select_atoms import SelectAtoms

        block.initialize(3)
        all_sel = SelectAtoms(seg_id="ADP")
        block.call(1, all_sel)
        assignments = block.get_block_assignments()
        assert 1 in assignments
        block.end()

    def test_call_with_string_selection(self):
        block.initialize(3)
        block.call(2, "type N .or. type CA .or. type C .or. type O")
        assignments = block.get_block_assignments()
        assert 2 in assignments
        block.end()

    def test_lambda_dynamics_setup(self):
        block.initialize(4)
        block.enable_lambda_dynamics()
        assert block._state.lambda_dynamics["enabled"]
        block.set_langevin(temp=300.0)
        for i in range(1, 5):
            block.ldin(
                i, lambda_sq=0.25, velocity=0.0, mass=12.0, bias=0.0, friction=50.0, ph_mode="none"
            )
        params = block.get_ldin_params(1)
        assert params is not None
        assert params["lambda_sq"] == pytest.approx(0.25, abs=1e-7)
        block.end()

    def test_langevin_coupling(self):
        block.initialize(3)
        block.enable_lambda_dynamics()
        block.set_langevin(temp=300.0)
        for i in range(1, 4):
            block.ldin(
                i, lambda_sq=0.25, velocity=0.0, mass=12.0, bias=0.0, friction=50.0, ph_mode="none"
            )
        assert block._state.langevin_enabled
        assert block._state.lambda_dynamics["langevin_temp"] == pytest.approx(300.0, abs=1e-7)
        block.disable_langevin()
        assert not block._state.langevin_enabled
        block.end()

    def test_bias_management(self):
        block.initialize(4)
        block.enable_lambda_dynamics()
        block.add_bias(block_i=1, block_j=2, cls=1, ref=0.5, cforce=5.0, npower=2)
        block.add_bias(block_i=2, block_j=3, cls=1, ref=0.5, cforce=5.0, npower=2)
        biases = block.get_biases()
        assert len(biases) == 2
        assert biases[0]["index"] == 1
        assert biases[1]["index"] == 2
        block.remove_bias(1)
        biases = block.get_biases()
        assert len(biases) == 1
        assert biases[0]["index"] == 1
        block.end()

    def test_msld_setup(self):
        block.initialize(5)
        block.enable_lambda_dynamics(theta=True)
        block.set_langevin(temp=300.0)
        for i in range(1, 6):
            block.ldin(
                i, lambda_sq=0.2, velocity=0.0, mass=12.0, bias=0.0, friction=50.0, ph_mode="none"
            )
        block.msld(site_assignments=[0, 1, 1, 2, 2], fnex=5.5)
        assert block._state.msld["enabled"]
        assert block._state.msld["fnex"] == pytest.approx(5.5, abs=1e-7)
        msld_state = block.get_msld_state()
        assert msld_state["site_assignments"][2] == 1
        assert msld_state["site_assignments"][4] == 2
        block.end()

    def test_soft_core(self):
        block.initialize(3)
        block.enable_lambda_dynamics()
        block.set_langevin(temp=300.0)
        for i in range(1, 4):
            block.ldin(
                i, lambda_sq=0.25, velocity=0.0, mass=12.0, bias=0.0, friction=50.0, ph_mode="none"
            )
        block.soft_core(mode="on")
        state = block.get_soft_core_state()
        assert state["mode"] == "on"
        block.soft_core(mode="off")
        state = block.get_soft_core_state()
        assert state["mode"] == "off"
        block.end()

    def test_rmla(self):
        block.initialize(3)
        block.enable_lambda_dynamics()
        block.set_langevin(temp=300.0)
        for i in range(1, 4):
            block.ldin(
                i, lambda_sq=0.25, velocity=0.0, mass=12.0, bias=0.0, friction=50.0, ph_mode="none"
            )
        block.rmla("bond", "theta")
        rmla_terms = block._state.lambda_dynamics.get("rmla_terms", [])
        assert "bond" in rmla_terms
        assert "theta" in rmla_terms
        block.end()

    def test_exclusion(self):
        block.initialize(4)
        block.add_exclusion((1, 2), (3, 4))
        exclusions = block._state.exclusions
        assert (1, 2) in exclusions
        assert (3, 4) in exclusions
        block.end()

    def test_direct_memory_access(self):
        block.initialize(3)
        block.coef(1, 2, 0.5)
        block.coef(2, 3, 0.7)
        block.end()
        avail = block.is_direct_access_available()
        matrix = block.get_coefficient_matrix(direct=True)
        if matrix is not None:
            assert matrix.shape == (3, 3)
            assert matrix[0, 1] == pytest.approx(0.5, abs=1e-7)
            assert matrix[1, 2] == pytest.approx(0.7, abs=1e-7)
        if avail.get("blockdata", False):
            lambdas = block.get_lambda_values_direct()
            if lambdas is not None:
                assert len(lambdas) == 3
            temp = block.get_temperature_direct()
            assert temp is None or isinstance(temp, float)

    def test_sync_state(self):
        block.initialize(3)
        block.coef(1, 2, 0.6)
        block.coef(2, 3, 0.4)
        block.end()
        result = block.sync_state_from_charmm()
        assert isinstance(result, bool)

    def test_modify_context_manager(self):
        block.initialize(3)
        block.coef(1, 2, 0.5)
        block.end()
        assert block.get_coefficient(1, 2) == pytest.approx(0.5, abs=1e-7)
        with block.modify():
            block.coef(1, 2, 0.8)
            block.coef(2, 3, 0.3)
        assert block.get_coefficient(1, 2) == pytest.approx(0.8, abs=1e-7)
        assert block.get_coefficient(2, 3) == pytest.approx(0.3, abs=1e-7)

    def test_modify_raises_without_init(self):
        block.clear()
        with pytest.raises(ValueError):
            with block.modify():
                block.coef(1, 2, 0.5)

    def test_set_coefficient_live(self):
        block.initialize(3)
        block.coef(1, 2, 0.5)
        block.end()
        result = block.set_coefficient_live(1, 2, 0.9)
        assert result
        assert block.get_coefficient(1, 2) == pytest.approx(0.9, abs=1e-7)

    def test_charmm_initialized_flag(self):
        block.clear()
        assert not block._state.charmm_initialized
        block.initialize(3)
        assert not block._state.charmm_initialized
        block.end()
        assert block._state.charmm_initialized

    def test_dual_topology_setup(self):
        block.clear()
        block.setup_dual_topology(
            reactant_selection="type CAY .or. type CY .or. type OY",
            product_selection="type NT .or. type CAT",
            lambda_value=0.5,
        )
        assert block.is_active()
        assert block.get_nblocks() == 3
        assert block.get_lambda() == pytest.approx(0.5, abs=1e-7)
        block.clear()

    def test_charmm_direct_read(self):
        block.clear()
        block.initialize(3)
        block.coef(1, 2, 0.7)
        block.coef(1, 3, 0.5)
        block.coef(2, 3, 0.3)
        block.end()
        charmm_val = block.get_coefficient(1, 2, direct=True)
        assert charmm_val is not None
        assert charmm_val == pytest.approx(0.7, abs=1e-5)
        charmm_val = block.get_coefficient(2, 3, direct=True)
        assert charmm_val is not None
        assert charmm_val == pytest.approx(0.3, abs=1e-5)
        charmm_val = block.get_coefficient(1, 1, direct=True)
        assert charmm_val is not None
        assert charmm_val == pytest.approx(1.0, abs=1e-5)
        cache_val = block.get_coefficient(1, 2, direct=False)
        assert cache_val == pytest.approx(0.7, abs=1e-5)
        block.clear()

    def test_charmm_matrix_read(self):
        block.clear()
        block.initialize(3)
        block.coef(1, 2, 0.8)
        block.coef(1, 3, 0.6)
        block.coef(2, 3, 0.4)
        block.end()
        matrix = block.get_coefficient_matrix(direct=True)
        assert matrix is not None
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert matrix[1, 1] == pytest.approx(1.0, abs=1e-5)
        assert matrix[2, 2] == pytest.approx(1.0, abs=1e-5)
        assert matrix[0, 1] == pytest.approx(0.8, abs=1e-5)
        assert matrix[1, 0] == pytest.approx(0.8, abs=1e-5)
        assert matrix[0, 2] == pytest.approx(0.6, abs=1e-5)
        assert matrix[1, 2] == pytest.approx(0.4, abs=1e-5)
        cache_matrix = block.get_coefficient_matrix(direct=False)
        assert cache_matrix is not None
        assert cache_matrix.shape == (3, 3)
        assert cache_matrix[0, 1] == pytest.approx(0.8, abs=1e-5)
        block.clear()

    def test_verify_coefficients_with_charmm(self):
        block.clear()
        block.initialize(3)
        block.coef(1, 2, 0.75)
        block.coef(2, 3, 0.55)
        block.end()
        result = block.verify_coefficients_with_charmm()
        assert result["match"], f"Verification failed: {result['differences']}"
        assert (1, 1) in result["python_cache"]
        assert (1, 1) in result["charmm_values"]
        assert len(result["differences"]) == 0
        block.clear()

    def test_is_active_unified(self):
        block.clear()
        active = block.is_active(direct=True)
        assert active in [False, None]
        active_cache = block.is_active(direct=False)
        assert not active_cache
        block.initialize(3)
        block.end()
        active = block.is_active(direct=True)
        assert active
        active_cache = block.is_active(direct=False)
        assert active_cache
        block.clear()
        active = block.is_active(direct=True)
        assert active in [False, None]
        active_cache = block.is_active(direct=False)
        assert not active_cache

    def test_python_charmm_sync_after_modify(self):
        block.clear()
        block.initialize(3)
        block.coef(1, 2, 0.5)
        block.end()
        with block.modify():
            block.coef(1, 2, 0.9)
        py_val = block.get_coefficient(1, 2, direct=False)
        charmm_val = block.get_coefficient(1, 2, direct=True)
        assert py_val == pytest.approx(0.9, abs=1e-5)
        assert charmm_val is not None
        assert charmm_val == pytest.approx(0.9, abs=1e-5)
        result = block.verify_coefficients_with_charmm()
        assert result["match"]
        block.clear()
