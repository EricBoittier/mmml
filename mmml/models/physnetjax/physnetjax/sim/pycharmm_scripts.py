adaptive_umbrella_script = """
    ! define the phi and chi1 dihedral angle as the two umbrella coordinates
umbrella dihe nresol 36 trig  6 poly 1 pept 1 C  pept 1 CA pept 1 N pept 1 CY
umbrella dihe nresol 36 trig  6 poly 1 pept 1 NT pept 1 C  pept 1 CA pept 1 N CA  

umbrella init nsim 1 update 100 equi 100 thresh 10 temp 300 -
              ucun 10 wuni 11
              """

Fcons = "1 cy 1 n 1 ca 1 c"
Ycons = "1 n 1 ca 1 c 1 nt"
cons_command = "cons dihe {} force {} min {:4.2f}'".format(
    Fcons, 2, -100.0
)  # "16 14 8 6"
# pycharmm.lingo.charmm_script(cons_command)


def add_waters(n_waters: int = 4):
    add_water_script = f"""! Generate a water segment
read sequence tip3 1
generate WAT setup angle 109.47
ic param
ic build
"""
    pycharmm.lingo.charmm_script(add_water_script)
    # minimize the water segment
    minimize.run_sd(**{"nstep": 10000, "tolenr": 1e-5, "tolgrd": 1e-5})
