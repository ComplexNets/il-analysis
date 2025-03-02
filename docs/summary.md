Ionic Liquid Disassembly
This program looks at the name of an ionic liquid and breaks it down into it's fragments.
This breakdown will be result in a cation, anion, alkyl group and functional group, though not all of these fragments may be present in the ionic liquid

The program takes the il_prop spreadsheet, breaks out the IL into it's fragment parts, as seen in the example spreadsheet. An example of the input that created the il_prop example is longList_frag.py.

The miniconda environment is here: C:\Users\X1\miniconda3\envs\analysis

streamlit>=1.24.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.13.0
scikit-learn>=1.2.0
scipy>=1.10.0
openpyxl>=3.1.0
rdkit
seaborn