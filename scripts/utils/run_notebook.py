import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_filename = 'NZEB_PIPELINE_ICAE2026.ipynb'

with open(notebook_filename, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
try:
    print("Executing notebook...")
    ep.preprocess(nb, {'metadata': {'path': './'}})
    print("Notebook executed successfully.")
except Exception as e:
    print(f"Error executing notebook: {e}")

with open(notebook_filename, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    
print("Notebook saved.")
