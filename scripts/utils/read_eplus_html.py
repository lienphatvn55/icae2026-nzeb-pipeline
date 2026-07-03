import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from html.parser import HTMLParser

html_file = r'ASHRAE901_OfficeMedium_STD2019_HoChiMinh.table.htm'

class TableExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row = []
        self.tables = []
        self.current_table = []
        self.table_names = []
        self.current_name = ""
        self.in_b = False
        
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
            self.current_table = []
        elif tag == 'tr':
            self.in_row = True
            self.current_row = []
        elif tag in ('td', 'th'):
            self.in_cell = True
        elif tag == 'b':
            self.in_b = True
            
    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)
                self.table_names.append(self.current_name)
            self.current_table = []
        elif tag == 'tr':
            self.in_row = False
            if self.current_row:
                self.current_table.append(self.current_row)
        elif tag in ('td', 'th'):
            self.in_cell = False
        elif tag == 'b':
            self.in_b = False
            
    def handle_data(self, data):
        text = data.strip()
        if self.in_cell and text:
            self.current_row.append(text)
        if self.in_b and text and not self.in_table:
            self.current_name = text

with open(html_file, 'r', encoding='utf-8') as f:
    content = f.read()

parser = TableExtractor()
parser.feed(content)

# Find specific important tables
targets = [
    'Site and Source Energy',
    'Site to Source Energy Conversion Factors',
    'End Uses',
    'End Uses By Subcategory',
    'Zone Information',
    'Heating Coils',
    'DX Cooling Coils',
    'Fans',
]

for target in targets:
    for i, name in enumerate(parser.table_names):
        if target.lower() in name.lower():
            table = parser.tables[i]
            print(f"\n{'='*80}")
            print(f"TABLE [{i}]: {name}")
            print(f"{'='*80}")
            for row in table[:25]:
                print("  | ".join(str(c)[:40] for c in row))
            if len(table) > 25:
                print(f"  ... ({len(table)} rows total)")
            break
