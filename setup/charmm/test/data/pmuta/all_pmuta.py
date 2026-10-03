#! /usr/bin/python
# probably working just for version 3

import sys,os,subprocess,time

comm_run=[]

# there is no HIS, but there are HSD,HSE,HSP

A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HSD':'H','HSE':'J','HSP':'O','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

aminos=[key for key,value in A1.items() ]

flag_first=True
for a in aminos:
    for b in aminos:
        if flag_first: # first one has a header...
            comm='pmuta.py '+a+' '+b+' h > '+a+b+'.out'
            flag_first=False
        else:
            comm='pmuta.py '+a+' '+b+' > '+a+b+'.out'
        print(comm)
        comm_run.append(subprocess.Popen(comm,shell=True))
for pp in comm_run: pp.wait() # this could go into a-loop, but the machine desn't die on being here

