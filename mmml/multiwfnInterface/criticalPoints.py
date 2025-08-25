import re
from collections import defaultdict

def parse_cp_output(text):
    blocks = re.split(r'-{10,}\s+CP\s+\d+,\s+Type\s+\([^)]+\)\s+-{10,}', text)
    headers = re.findall(r'-{10,}\s+CP\s+(\d+),\s+Type\s+\(([^)]+)\)\s+-{10,}', text)

    data = []

    for header, block in zip(headers, blocks[1:]):  # blocks[0] is before first CP
        cp_index, cp_type = header
        cp_data = {
            "CP_index": int(cp_index),
            "Type": cp_type.strip(),
            "Scalars": {},
            "Vectors": {},
            "Matrices": {},
            "Eigenvalues": {},
            "Eigenvectors": {}
        }

        lines = block.strip().split('\n')

        # Helper functions
        def extract_vector(label, line):
            values = re.findall(r'[-+]?\d*\.\d+E[+-]\d+|\d+\.\d+', line)
            return list(map(float, values)) if values else None

        def extract_scalar(label, line):
            match = re.search(rf'{re.escape(label)}.*?:\s+([-\d.E+]+)', line)
            return float(match.group(1)) if match else None

        # Parsing logic
        i = 0
        while i < len(lines):
            line = lines[i]

            if "Corresponding nucleus" in line:
                match = re.search(r'Corresponding nucleus:\s+(\d+)\((.*?)\)', line)
                if match:
                    cp_data["Nucleus"] = {"Index": int(match.group(1)), "Element": match.group(2).strip()}

            elif "Position (Bohr)" in line:
                cp_data["Vectors"]["Position_Bohr"] = extract_vector("Position (Bohr)", line)

            elif "Position (Angstrom)" in line:
                cp_data["Vectors"]["Position_Angstrom"] = extract_vector("Position (Angstrom)", line)

            elif "Density of all electrons" in line:
                cp_data["Scalars"]["Density_all"] = extract_scalar("Density of all electrons", line)

            elif "G(r) in X,Y,Z" in line:
                cp_data["Vectors"]["G_r_components"] = extract_vector("G(r) in X,Y,Z", line)

            elif "Components of gradient" in line:
                cp_data["Vectors"]["Gradient"] = extract_vector("Components of gradient", lines[i+1])
                i += 1

            elif "Components of Laplacian" in line:
                cp_data["Vectors"]["Laplacian_components"] = extract_vector("Components of Laplacian", lines[i+1])
                i += 1

            elif "Hessian matrix:" in line:
                matrix = []
                for j in range(1, 4):
                    matrix.append(extract_vector("", lines[i + j]))
                cp_data["Matrices"]["Hessian"] = matrix
                i += 3

            elif "Eigenvalues of Hessian" in line:
                cp_data["Eigenvalues"]["Hessian"] = extract_vector("Eigenvalues of Hessian", line)

            elif "Eigenvectors (columns) of Hessian" in line:
                eigenvectors = []
                for j in range(1, 4):
                    eigenvectors.append(extract_vector("", lines[i + j]))
                cp_data["Eigenvectors"]["Hessian"] = eigenvectors
                i += 3

            elif "Stress tensor:" in line:
                matrix = []
                for j in range(1, 4):
                    matrix.append(extract_vector("", lines[i + j]))
                cp_data["Matrices"]["Stress_tensor"] = matrix
                i += 3

            elif "Eigenvalues of stress tensor" in line:
                cp_data["Eigenvalues"]["Stress_tensor"] = extract_vector("Eigenvalues of stress tensor", line)

            elif "Eigenvectors (columns) of stress tensor" in line:
                eigenvectors = []
                for j in range(1, 4):
                    eigenvectors.append(extract_vector("", lines[i + j]))
                cp_data["Eigenvectors"]["Stress_tensor"] = eigenvectors
                i += 3

            else:
                scalar_match = re.search(r'^(.*?)\s*:\s*([-\d.E+]+)$', line.strip())
                if scalar_match:
                    label = scalar_match.group(1).strip()
                    value = float(scalar_match.group(2))
                    cp_data["Scalars"][label] = value

            i += 1

        data.append(cp_data)

    return data



text = open("/home/boittier/Multiwfn_3.8_dev_bin_Linux/CPprop.txt").read()
data = parse_cp_output(text)


# In[55]:


text


# In[56]:


import pandas as pd


# In[57]:


pd.DataFrame(data)

pd.DataFrame([_["Scalars"] for _ in data])