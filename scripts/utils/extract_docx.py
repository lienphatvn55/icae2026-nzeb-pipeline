import zipfile
import xml.etree.ElementTree as ET
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

filepath = os.path.join(os.path.dirname(__file__), 'CẬP NHẬT TIẾN ĐỘ ICAE 2026.07.01.docx')
z = zipfile.ZipFile(filepath)
tree = ET.parse(z.open('word/document.xml'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

output = []
for p in tree.findall('.//w:p', ns):
    texts = []
    for t in p.findall('.//w:t', ns):
        if t.text:
            texts.append(t.text)
    if texts:
        output.append(''.join(texts))

full_text = '\n'.join(output)

# Save to text file
with open('progress_doc.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"Total lines: {len(output)}")
print("File saved to progress_doc.txt")
