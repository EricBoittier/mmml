# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Class for CHARMM implicit solvent modules. 

Corresponds to CHARMM module `FACTS` for *Fast Analytical Continuum Treatment of Solvation*  
See [FACTS documentation](https://academiccharmm.org/documentation/latest/facts)  

Created by Yujin Wu | wyujin@umich.edu
 
"""

import pycharmm
from pycharmm.script import CommandScript

class FACTS(CommandScript):
    def __init__(self, **kwargs):
        super().__init__('FACTS', **kwargs)

