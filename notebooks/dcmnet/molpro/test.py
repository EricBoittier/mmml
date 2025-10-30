
from pymolpro import Project
from pathlib import Path
Path(".").glob("*xml")
list(Path(".").glob("*xml"))

from pymolpro import Project
from pathlib import Path
list(Path(".").glob("*xml"))
xmlfiles = list(Path(".").glob("*xml"))
p = Project(xmlfiles[0])
p = Project(xmlfiles[0].to_string())
p = Project(str(xmlfiles[0]))
p
dir(p)
p.variables
p.variables()
p.view_input()
p.variable
p.variable()
p.variable("DIPX")

p = Project(str(xmlfiles[0][:-4]))
p = Project(str(xmlfiles[0])[:-4])
p.variable("DIPX")

from pymolpro import Project
p = Project(files=str(xmlfiles[0])[:-4])
from pathlib import Path
xmlfiles = list(Path(".").glob("*xml"))
p = Project(files=str(xmlfiles[0])[:-4])
p
p.variable("DIPX")
p = Project(files=str(xmlfiles[0]))
p.variable("DIPX")
p = Project(files=[str(xmlfiles[0])])
p.variable("DIPX")
dir(p)
p.xyz
p.properties()
p.property_names()
p.property_names('property_hasg')
p.property('property_hash')
p.properties()
p.properties_old()
p.properties_old()[0]
dir(p.properties_old()[0])
dict(p.properties_old()[0])
p.properties_old()[0].tree_view()

p = Project(files=["co2_r1_1p10_r2_1p25_ang_160.xml"])
print(type(p.geometries), callable(getattr(p,'geometries',None)))
gi = p.geometries()[0] if callable(p.geometries) else p.geometries[0]
print(type(gi))
if isinstance(gi, dict):
    print(gi.keys())
    first = next(iter(gi.values()))
    print('first value type:', type(first))
print(gi)

# # from test import *
# # _parse_molpro_out("co2_r1_1p10_r2_1p30_ang_120.out")
# # exit()
# import lxml import etree
# from lxml import etree
# tree = etree.parse("co2_r1_1p25_r2_1p25_ang_180.xml")
# exit()
