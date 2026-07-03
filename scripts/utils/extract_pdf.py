import fitz  # PyMuPDF
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

filepath = os.path.join(os.path.dirname(__file__), '2025_BEM with replaceable components using graph attention networks.pdf')
doc = fitz.open(filepath)

output_file = 'reference_paper.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc):
        text = page.get_text()
        f.write(f"\n{'='*80}\nPAGE {i+1}\n{'='*80}\n")
        f.write(text)

print(f"Total pages: {len(doc)}")
print(f"Saved to {output_file}")
