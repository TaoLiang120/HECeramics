import os
import pandas as pd

Element_file = "element.csv"
DF_elements = pd.read_csv(os.path.join(os.path.dirname(__file__), Element_file))
DF_elements = DF_elements.set_index("Composition")

Binary_file = "rocksalt_binary.csv"
DF_binaries = pd.read_csv(os.path.join(os.path.dirname(__file__), Binary_file))
DF_binaries = DF_binaries.set_index("Composition")

def get_DF_Index(style, a, b=None, c=None, d=None):
    if style == 0:
        return a
    elif style == 1:
        return a + "C"
    elif style == 2:
        return a + "O2"
    elif style == 3:
        return a + "2" + b + "2" + "O7"