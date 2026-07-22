#!/bin/bash
cd /mmhome/boittier/home/mmml
source .venv/bin/activate
mmml orbax-to-json artifacts/spooky_so3lr_muon3/epoch-0013 -o spooky_so3lr_muon3_epoch0013.json
