import os
import polars as pl

#DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy  TEMPerature
#DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor        VIRKe
#DYNA INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals    IMPRopers
#DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP         USER
#DYNA IMAGES:        IMNBvdw       IMELec       IMHBnd       RXNField    EXTElec
#DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI       VOLUme
#DYNA>        0      0.00000 -61671.83101    283.74974 -61955.58076    124.11071
#DYNA PRESS>       418.64436   -413.41713  -3099.63460  -1660.34713   9261.00000

def read_dyna_line(line: str):
    dyna = None
    step = None
    time = None
    total_energy = None
    total_kinetic_energy = None
    energy = None
    temperature = None
    
    if line.startswith("DYNA"):
        dyna = True
    else:
        raise ValueError(f"Line {line} does not start with DYNA")

    step = int(line[5:18])
    time = float(line[19:28])
    total_energy = float(line[34:44])
    total_kinetic_energy = float(line[44:54])
    energy = float(line[54:64])
    temperature = float(line[71:74])
    output = {
        "dyna": dyna,
        "step": step,
        "time": time,
        "total_energy": total_energy,
        "total_kinetic_energy": total_kinetic_energy,
        "energy": energy,
        "temperature": temperature
    }
    return output 

def read_open_line(line: str):
    pass


def read_press_line(line: str):
    vire = None
    viri = None
    press_e = None
    press_i = None
    volume = None
    if line.startswith("DYNA PRESS"):
        vire = float(line[11:28])
        viri = float(line[30:40])
        press_e = float(line[30:42])
        press_i = float(line[42:55])
        volume = float(line[68:80])
    output = {
        "vire": vire,
        "viri": viri,
        "press_e": press_e,
        "press_i": press_i,
        "volume": volume
    }
    return output

if __name__ == "__main__":
    test_dyna_line = "DYNA PRESS>       418.64436   -413.41713  -3099.63460  -1660.34713   9261.00000"
    print(read_press_line(test_dyna_line))
    test_dyna_line = "DYNA>        0      0.00000 -61671.83101    283.74974 -61955.58076    124.11071" 
    print(read_dyna_line(test_dyna_line))

    from pathlib import Path
    current_dir = Path(__file__).parent
    dyna_file = current_dir / ".." / ".." / "testdata" / "DYNA1"
    press_file = current_dir / ".." / ".." / "testdata" / "PRESS1"

    dyna_data = []
    press_data = []

    for line in open(dyna_file):
        if line.startswith("DYNA"):
            dyna_data.append(read_dyna_line(line))
    
    for line in open(press_file):   
        if line.startswith("DYNA PRESS"):
            press_data.append(read_press_line(line))

    dyna_df = pl.DataFrame(dyna_data)
    press_df = pl.DataFrame(press_data)

    n_steps = len(dyna_df)
    n_press = len(press_df)

    print(dyna_df)
    dyna_df.write_csv("dyna.csv")

    print(press_df) 
    press_df.write_csv("press.csv")

