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
    dyna, step, time, total_energy, total_kinetic_energy, energy, temperature = None
    if line.startswith("DYNA"):
        dyna = True
    else:
        raise ValueError(f"Line {line} does not start with DYNA")

    step = int(line[5:10])
    time = float(line[10:20])
    total_energy = float(line[20:30])
    total_kinetic_energy = float(line[30:40])
    energy = float(line[40:50])
    temperature = float(line[50:60])
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
    vire, viri, press_e, press_i, volume = None
    if line.startswith("DYNA PRESS"):
        vire = float(line[10:20])
        viri = float(line[20:30])
        press_e = float(line[30:40])
        press_i = float(line[40:50])
        volume = float(line[50:60])
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
    test_open_line = "DYNA PROP>       124.11071   -413.41713  -3099.63460  -1660.34713   9261.00000"
    print(read_open_line(test_open_line))
    test_prop_line = "DYNA PROP>       124.11071   -413.41713  -3099.63460  -1660.34713   9261.00000"
    print(read_prop_line(test_prop_line))

