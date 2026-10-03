#! /usr/bin/python
# probably working just for version 3

import sys,os,subprocess,time

# there is no HIS, but there are HSD,HSE,HSP

built={ 'ALA':45.8198954 ,'ARG':-78.2998568 ,'ASN':106.394081 ,
        'ASP':60.3806192 , 'CYS':54.3995198,'GLU':56.4892554
        ,'GLN':35.6003248 , 'GLY':35.1762181,'HSD':74.1222949
        ,'HSE':31.0928671 , 'HSP':120.6731,'ILE':62.9521772 ,
        'LEU':15667.855 , 'LYS':58.5139161,'MET':54.5110815 ,
        'PHE':80.3909839 , 'PRO':1422.47518,'SER':55.0518984
        ,'THR':43.5340424 , 'TRP':83.6107377,'TYR':74.9184013
        ,'VAL':145.878499 }

# minimized structures: not yet ??? these are for mono aminos
minim={'ALA': 17.4231178 ,'ARG':-108.003241 ,'ASN': -16.1152685,
       'ASP':-0.714726904,'CYS':  23.19704  ,'GLU':  12.5160616,'GLN': -0.451706664,
       'GLY': 12.4895571 ,'HSD':  18.3572967,'ILE':  24.3682154,
       'LEU': 14.7685489 ,'LYS':  34.0977693,'MET':  24.0893864,
       'PHE': 36.4384842 ,'PRO':  24.2108489,'SER':  24.4483924,'THR': 12.2968822,
       'TRP': 47.0407731 ,'TYR':  28.8080799,'VAL':  19.6981279}

A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HSD':'H','HSE':'J','HSP':'O','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

aminos=[key for key in A1.keys()]

cexe='~/charmm/building/c50a2/charmm'
gen='gen_pmuta_test'
geninp=gen+'.inp'
for a in aminos:
    comm_run=[]
    for b in aminos:
        patch_name='x'+A1[a]+'2'+A1[b]
        reverse_patch='x'+A1[b]+'2'+A1[a]
        comm= cexe+' -i  '+geninp+' wt='+a+' p='+patch_name+' r='+reverse_patch
        comm += ' res='+'{0:f}'.format(built[b])+' mr='+b
        comm += ' > '+gen+'_'+patch_name+'.out'
        print(comm)
        comm_run.append(subprocess.Popen(comm,shell=True))
    for pp in comm_run: pp.wait() # this could go into a-loop
#    break
    #get_res='grep RESULT: gen_muta_test_'+patch_name+'.out'
    #os.system(get_res)
