# Same workflow in a notebook: open 01_make.ipynb and run the cells (no settings.source needed).
source settings.source
$PY"/make_res.py" --res $RES > make.res.out

$PY"/make_box.py" --res $RES --n $N --side_length $L > make.box.out



