#! /usr/bin/python
# probably working just for version 3

import sys

'''

INVOKE this program
(it needs a toppar to be linked to a real toppar in charmm tree):

./pmuta.py <from> <to>  [h]

3 letter names need to be specified for <from> <to>
case does not matter!
h is optional to put a header into the .rtf file
When creating many mutations only first one needs a header

Maybe this program may be generalized for patches to modify
any residue into any other


Let's do the mutations in an automatic way.

What is wrong with mkpres:
1. I am not sure yet - issues with the energies in the test cases
   /see NOTES.mutations/
2. It is not really fully automated

So we read toppar/top_all36_prot.rtf file
And from there create a patch residue:

pres <name> <tot_charge>

put it into a file <name>.rtf
read it into a PERT input script like this:

setup WT...

read rtf card append name <name>.rtf

PERT
patch <name> <site_num> <site_left> <site_right> - it needs 3 arguments!!
.... there can be even more patches ... for multi - site mutations


We are starting with the muta.py program - this means dual topology ??
(Boresch papers disagree: Figure 2 vs Figure 3 - which one is correct???)
[Let's assume Figure 3 definition is correct]


first test could be:

setup X, calc energy
patch X2Y, calc energy
patch Y2X, calc energy (is it the same as X?)

second test would be:

script 1: setup X, calc energy
script 2: setup Y, calc energy
script 3:
  - setup X 
  - pert
  - patch X2Y
     needs:
      - delete old atoms which are not the same in the new residue (backbone stays)
      - delete also all the rtf definitions: bond,impropers,cmap
      - specify new atoms
      - specify new bonds,impropers,cmap
  - energy (lambda=0), compare to script 1
  - energy (lambda=1), compare to script 2

NOTES:
1. do we worry about mutating cter, nter ???

==========================================

But later we can also do the dummies in a mutad.py & pmutad.py programs

'''

# define/use the tables for 3 letter names and 1 letter names
# se we can write PRES <char_from_res1>2<char_from_res2>
# because of name clashes with the rest of charmm/toppar names we prepend 'X'
# so the names are PRES x<res1>2<res2>  res1,res2 are one letter names: hsd=h,hse=j,hdp=o

# amino acid names: NOTE!!! his=hsd here, but hse and hsp are J and O 
A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HSD':'H','HSE':'J','HSP':'O','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

A3={ value:key for key,value in A1.items() }


# this is where all amino acids reside
#tops=open('toppar/top_all36_na.rtf',mode='rt')
#tops=open('toppar/top_all36_cgenff.rtf',mode='rt')
tops=open('toppar/top_all36_prot.rtf',mode='rt')

# locate an AA:

#print(sys.argv[1])

list_residues = False
if len(sys.argv) == 1:
    list_residues = True ;

if ( len(sys.argv) < 3 ) and ( not list_residues ) :
    print('Missing argument!!!')
    exit(-1)

process_res1 = False ; process_res2 = False
nam1=[] ; typ1=[] ; chg1=[] ; group1=[] ; bond1=[]
double1=[] ; improper1=[] ; cmap1=[] ; donor1=[]
acceptor1=[] ; ic1=[]
nam2=[] ; typ2=[] ; chg2=[] ; group2=[] ; bond2=[]
double2=[] ; improper2=[] ; cmap2=[] ; donor2=[]
acceptor2=[] ; ic2=[]

for resi in tops:
    curlin=resi.split()
    if len(curlin) > 0:
        if process_res1:
            # lets take care of atoms first... We'll get to the groups later...
            if curlin[0].casefold().find('atom'.casefold()) > -1 :
                # we have an atom - collect the info...
                nam1.append(curlin[1])
                typ1.append(curlin[2])
                chg1.append(curlin[3])
            if curlin[0].casefold().find('grou'.casefold()) > -1 :
                group1.append(len(nam1))
            if curlin[0].casefold().find('bond'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    bond1.append(curlin[g:g+2])
            if curlin[0].casefold().find('doub'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    double1.append(curlin[g:g+2])
            if curlin[0].casefold().find('impr'.casefold()) > -1 :
                for g in range(1,len(curlin),4):
                    improper1.append(curlin[g:g+4])
            if curlin[0].casefold().find('cmap'.casefold()) > -1 :
                for g in range(1,len(curlin),8):
                    cmap1.append(curlin[g:g+8])
            if curlin[0].casefold().find('donor'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    donor1.append(curlin[g:g+2])
            # it can actually be only one atom specified for acceptor, but it still works like this...
            if curlin[0].casefold().find('acce'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    acceptor1.append(curlin[g:g+2])
            if curlin[0].casefold().find('ic'.casefold()) > -1 :
                ic1.append(curlin[1:])


            # process this residue until next residue comes...
            if curlin[0].casefold().find('resi'.casefold()) > -1:
                process_res1 = False
            
        if process_res2:
            # lets take care of atoms first... We'll get to the groups later...
            if resi.split()[0].casefold().find('atom'.casefold()) > -1 :
                # we have an atom - collect the info...
                nam2.append(resi.split()[1])
                typ2.append(resi.split()[2])
                chg2.append(resi.split()[3])
            if curlin[0].casefold().find('grou'.casefold()) > -1 :
                group2.append(len(nam2))
            if curlin[0].casefold().find('bond'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    bond2.append(curlin[g:g+2])
            if curlin[0].casefold().find('doub'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    double2.append(curlin[g:g+2])
            if curlin[0].casefold().find('impr'.casefold()) > -1 :
                for g in range(1,len(curlin),4):
                    improper2.append(curlin[g:g+4])
            if curlin[0].casefold().find('cmap'.casefold()) > -1 :
                for g in range(1,len(curlin),8):
                    cmap2.append(curlin[g:g+8])
            if curlin[0].casefold().find('donor'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    donor2.append(curlin[g:g+2])
            if curlin[0].casefold().find('acce'.casefold()) > -1 :
                for g in range(1,len(curlin),2):
                    acceptor2.append(curlin[g:g+2])
            if curlin[0].casefold().find('ic'.casefold()) > -1 :
                ic2.append(curlin[1:])


            # process until the next resi command...
            if resi.split()[0].casefold().find('resi'.casefold()) > -1:
                process_res2 = False

        # to avoid patch residues maybe > -1 should be == 0 ??
        if curlin[0].casefold().find('resi'.casefold()) > -1:
            if list_residues :
                print(curlin[1])
                continue # go find next residue command...
            # above continue makes the following 2 ifs being ignored 
            if curlin[1].casefold() == sys.argv[1].casefold() :
            # good for small single leters: if curlin[1].casefold() == A3[sys.argv[1].upper()].casefold() :
                print(resi.split()[1]+' was found')
                process_res1 = True
                res1_name=curlin[1]

            if curlin[1].casefold() == sys.argv[2].casefold() :
            #good for small single letters: if curlin[1].casefold() == A3[sys.argv[2].upper()].casefold() :
                print(resi.split()[1]+' was found')
                process_res2 = True
                res2_name=curlin[1]


if list_residues : exit(1)

atom_neighbors=['-CA','-C','-O','+N','+HN','+CA']
# -atom replaced with 2atom ; +atom replaced by 3atom
A = set(atom_neighbors)
print('resi 1:',A1[res1_name])
print(nam1,typ1,chg1,group1)
print('BONDS',bond1)
print('DOUBLE',double1)
print('IMPROPER',improper1)
print('CMAP',cmap1)
print('DONOR',donor1)
print('ACCEPTOR',acceptor1)
print('IC',ic1)
# just checking: for i in ic1 : print(i[6],end=' ')
print()

# checking printout
print('resi 2:',A1[res2_name])
print(nam2,typ2,chg2,group2)
print('BONDS',bond2)
print('DOUBLE',double2)
print('IMPROPER',improper2)
print('CMAP',cmap2)
print('DONOR',donor2)
print('ACCEPTOR',acceptor2)
print('IC',ic2)
# just checking: for i in ic2 : print(i[6],end=' ')
print()
# end of debug printing - now for the real patch residue

# write to a .rtf file:
# name clashes:
# so 3 letter ones are already in use, mostly in toppar_all36_na_rna_modified.str
# prepend x, we still want the same name for rtf as for pres
rtf=open('x'+A1[res1_name].lower()+'2'+A1[res2_name].lower()+'.rtf',mode='wt')
# see if we need a header file
if len(sys.argv) > 3:
    if sys.argv[3].casefold() == 'h'.casefold():
        print('* generated by ',sys.argv[0],' program',file=rtf)
        print('* there is a bug in CHARMM - not working:  delete cmap ',file=rtf) # CHARMM BUG
        print('* it is going to be fixed in c47a2 (maybe also c46b2?) ',file=rtf) # CHARMM BUG
        print('* NOTE: HSE is J and HSP is O in one letter names ',file=rtf) # NOTE for HSE,HSP
        print('* There are test cases and programs available at test/c50test ',file=rtf) # test cases
        print('* Use pmuta.py <from> <to> [h] ',file=rtf) # use pmuta.py
        print('* <from>,<to> case insensitive 3 letter names for AAs ',file=rtf) # use pmuta.py
        print('* the algorithm is maybe good to patch any residue to any other - UNTESTED! ',file=rtf) # suggestion
        print('* see test/c50test/pmuta.py,test/c50test/all_pmuta.py,test/c50test/all_pmuta_test.py ',file=rtf)
        print('*',file=rtf) # maybe more info here, eg: which mutation is in the file ??
        print('36  1',file=rtf) # always OK
        print(file=rtf) # just empty line for clarity....
tot_chg2=0.0
for i in chg2: tot_chg2 += float(i)
#printout for the RTF: one can use {2:>8s} for the right adjustement
print('PRES', 'X'+A1[res1_name]+'2'+A1[res2_name],'{0:8.4f}'.format(tot_chg2),file=rtf)
# first delete the atoms in nam1 that are not present in nam2: subtract the sets: nam1-nam2
s1=set(nam1) ; s2=set(nam2)
del_atoms = s1 - s2
#print('delete:',del_atoms)

# before deleting atoms we need to delete all the old stuff:
## bonds, doubles, impropers and cmap + donors, acceptors (although they don't contribute to forces)
### these are kept if atoms in them are not deleted
### and if their bond structure changes, they are worng to keep!!!!
#### DOESN'T WORK HERE - NEEDS TO IGNORE DELETED ATOMS

for ib in bond1:
    if not del_atoms.isdisjoint(ib) : continue  # only non-deleted atoms!!!
    bond_line='{0:11s}  {1:8s}  {2:8s}'.format('DELETE BOND',ib[0],ib[1])
    # no such things in PRES!! : if not A.isdisjoint(ib): bond_line = '! ' + bond_line
    bond_line=bond_line.replace('-','2')
    bond_line=bond_line.replace('+','3') ; 
    print(bond_line,file=rtf)

for ib in double1:
    if not del_atoms.isdisjoint(ib) : continue  # only non-deleted atoms!!!
    bond_line='{0:13s}  {1:8s}  {2:8s}'.format('DELETE DOUBLE',ib[0],ib[1])
    # nu such thing in PRES: if not A.isdisjoint(ib): bond_line = '! ' + bond_line
    bond_line=bond_line.replace('-','2')
    bond_line=bond_line.replace('+','3') ; 
    print(bond_line,file=rtf)

for im in improper1:
    if not del_atoms.isdisjoint(im) : continue  # only non-deleted atoms!!!
    impro_line = 'DELETE IMPROPER  '
    impro_line += '{0:8s} {1:8s} {2:8s} {3:8s}'.format(im[0],im[1],im[2],im[3])
    # if not A.isdisjoint(im) : impro_line = '! ' + impro_line
    impro_line=impro_line.replace('-','2')
    impro_line=impro_line.replace('+','3') ; 
    print(impro_line,file=rtf)

for cm in cmap1:
    if not del_atoms.isdisjoint(cm) : continue  # only non-deleted atoms!!!
    cmap_line = 'DELETE CMAP  '
    cmap_line += '{0:8s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s} {6:8s} {7:8s}'.format(
        cm[0],cm[1],cm[2],cm[3],cm[4],cm[5],cm[6],cm[7])
    # if not A.isdisjoint(cm) : cmap_line = '! ' + cmap_line
    cmap_line=cmap_line.replace('-','2')
    cmap_line=cmap_line.replace('+','3')
#CHARMM BUG: /no delete cmap/   print(cmap_line,file=rtf)

for delat in del_atoms: print('DELETE ATOM ',delat,file=rtf)
for iat in range(len(nam2)):
    if iat in group2: print('GROUP',file=rtf)
    print("{0:8s} {1:8s} {2:8s} {3:10.3f}".format('ATOM', nam2[iat],typ2[iat],float(chg2[iat])),file=rtf)
for ib in bond2:
    bond_line='{0:6s}  {1:8s}  {2:8s}'.format('BOND',ib[0],ib[1])
    #if not A.isdisjoint(ib): bond_line = '! ' + bond_line
    bond_line=bond_line.replace('-','2')
    bond_line=bond_line.replace('+','3') ; 
    print(bond_line,file=rtf)
for ib in double2:
    bond_line='{0:6s}  {1:8s}  {2:8s}'.format('DOUBLE',ib[0],ib[1])
    #if not A.isdisjoint(ib): bond_line = '! ' + bond_line
    bond_line=bond_line.replace('-','2')
    bond_line=bond_line.replace('+','3') ; 
    print(bond_line,file=rtf)
for im in improper2:
    impro_line = 'IMPROPER  '
    impro_line += '{0:8s} {1:8s} {2:8s} {3:8s}'.format(im[0],im[1],im[2],im[3])
    #if not A.isdisjoint(im) : impro_line = '! ' + impro_line
    impro_line=impro_line.replace('-','2')
    impro_line=impro_line.replace('+','3') ; 
    print(impro_line,file=rtf)
for cm in cmap2:
    cmap_line='{0:6s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s} {6:8s} {7:8s} {8:8s}'.format(
        'CMAP',cm[0],cm[1],cm[2],cm[3],cm[4],cm[5],cm[6],cm[7])
    #if not A.isdisjoint(cm) : cmap_line = '! ' + cmap_line
    cmap_line=cmap_line.replace('-','2')
    cmap_line=cmap_line.replace('+','3') ; 
    # CHARMM BUG: /no delete cmap!!!/ print(cmap_line,file=rtf)
for xic in ic2:
    ic_line='{0:6s} {1:8s} {2:8s} {3:8s} {4:8s}'.format('IC',xic[0],xic[1],xic[2],xic[3])
    ic_line=ic_line.replace('-','2')
    ic_line=ic_line.replace('+','3') ; 
    ic_line += '{0:10.4f} {1:10.4f} {2:10.4f} {3:10.4f} {4:10.4f}'.format(
        float(xic[4]),float(xic[5]),float(xic[6]),float(xic[7]),float(xic[8]))
    #if not A.isdisjoint(xic[:5]) : ic_line = '! ' + ic_line
    print(ic_line,file=rtf)
print(file=rtf)

#print(process_res1,process_res2)

#print(A1)
#print(A3)
