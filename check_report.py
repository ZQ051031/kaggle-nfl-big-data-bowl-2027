"""Enforce the competition writeup limits on the generated Markdown."""
import re
from pathlib import Path

def check(path='REPORT.md'):
    text=Path(path).read_text()
    words=len(re.findall(r'\S+',text))
    figures=len(re.findall(r'!\[',text))
    tables=len(re.findall(r'^\|\s*:?-{3}',text,flags=re.M))
    print(f'REPORT.md words: {words} (limit 2000)')
    print(f'Figures: {figures}; tables: {tables}; total: {figures+tables} (must be <10)')
    assert words<=2000 and figures+tables<10
    assert '{{' not in text

if __name__=='__main__': check()
