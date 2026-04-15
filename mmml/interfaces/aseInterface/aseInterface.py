import ase
import ase.calculators.calculator as ase_calc


class MessagePassingCalculator(ase_calc.Calculator):
    implemented_properties = ["energy", "forces", "dipole"]

    def calculate(
        self, atoms, properties, system_changes=ase.calculators.calculator.all_changes
    ):
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
        output = None
        self.results["energy"] = output[
            "energy"
        ].squeeze()  # * (ase.units.kcal/ase.units.mol)
        self.results["forces"] = output[
            "forces"
        ]  # * (ase.units.kcal/ase.units.mol) #/ase.units.Angstrom
