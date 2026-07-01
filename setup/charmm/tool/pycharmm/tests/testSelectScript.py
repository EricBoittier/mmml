from pycharmm import *

charmm_data_dir = 'data'
rtf_fn = charmm_data_dir + '/top_all36_prot.rtf'
chm_read.rtf(rtf_fn)

prm_fn = charmm_data_dir + '/par_all36_prot.prm'
chm_read.prm(prm_fn, flex=True)

# begin toppar/toppar_water_ions.str
chm_read.rtf('data/water_ions.rtf', append=True)
chm_read.prm('data/water_ions.prm', append=True, flex=True)

old_warn_level = settings.set_warn_level(-1)
old_bomb_level = settings.set_bomb_level(-1)

chm_read.prm('data/sodium_oxygen_nbfixes.prm', append=True, flex=True)

settings.set_warn_level(old_warn_level)
settings.set_bomb_level(old_bomb_level)
# end toppar/toppar_water_ions.str

chm_read.sequence_string('ALA')

gen.new_segment('ADP', 'ACE', 'CT3', setup_ic=True)

ic.prm_fill(False)
ic.seed(1, 'CAY', 1, 'CY', 1, 'N')
ic.build()

coor.orient()
coor.show()

my_nbonds = NonBondedScript(cutnb=18.0, ctonnb=15.0, ctofnb=13.0,
                                     eps=1.0, cdie=True, atom=True,
                                     vatom=True, fswitch=True, vfswitch=True)
my_nbonds.run()
charmm_script("define STUFF select all end")
chm_select.store_selection('HYD', chm_select.hydrogen())
chm_select.find('HYD')
charmm_script('coor stat sele STUFF end')
charmm_script('coor stat sele HYD end')

h_atoms = SelectAtoms().all_hydrogen_atoms()
h_atoms_name = h_atoms.store()

charmm_script('coor stat sele ' +
                    h_atoms_name +
                    ' end')

max_name = chm_select.get_max_name()
n_stored = chm_select.get_num_stored()

stored_selections = chm_select.get_stored_names()
print(stored_selections)

chm_select.delete_stored_selection('HYD')

stored_selections = chm_select.get_stored_names()
print(stored_selections)

h_atoms.unstore()

stored_selections = chm_select.get_stored_names()
print(stored_selections)

chm_select.delete_stored_selection('STUFF')

stored_selections = chm_select.get_stored_names()
print(stored_selections)

n_stored = chm_select.get_num_stored()

with SelectAtoms(hydrogens=True) as my_sel:
    name = my_sel.get_stored_name()
    charmm_script('coor stat sele {} end'.format(name))
    stored_selections = chm_select.get_stored_names()
    print(stored_selections)
    n_stored = chm_select.get_num_stored()
    print(n_stored)

stored_selections = chm_select.get_stored_names()
print(stored_selections)
n_stored = chm_select.get_num_stored()
print(n_stored)

chm_write.coor_pdb('testy.pdb',
                        selection=SelectAtoms(hydrogens=True))

my_nbonds_settings = NonBondedScript(selection=SelectAtoms(hydrogens=True))
my_nbonds_settings.run()

print('hydrogen atom selection stats')
print('atom indexes:')
print(h_atoms.get_atom_indexes())
print('chem types:')
print(h_atoms.get_chem_types())
print('residue indexes:')
print(h_atoms.get_res_indexes())
print('residue names:')
print(h_atoms.get_res_names())
print('residue ids:')
print(h_atoms.get_res_ids())
print('segment indexes:')
print(h_atoms.get_seg_indexes())
print('segment ids:')
print(h_atoms.get_seg_ids())
print('atom types:')
print(h_atoms.get_atom_types())

h_atoms.store('foo')
stored_selections = chm_select.get_stored_names()
print(stored_selections)
n_stored = chm_select.get_num_stored()
print(n_stored)

charmm_script('coor stat sele ' +
                    'foo' +
                    ' end')
