"""Consolidate extraction metadata and inspect review artifacts; never edit sources."""
import json
import hashlib
import re
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
REVIEW = WORK.parent
sys.stdout.reconfigure(encoding='utf-8')

def main():
    entries = {}
    for manifest in WORK.glob('清单_*.json'):
        for item in json.loads(manifest.read_text(encoding='utf-8')):
            if 'error' not in item:
                entries[item['id']] = item
    ordered = [entries[k] for k in sorted(entries)]
    (WORK / '有效来源清单.json').write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'sources': len(ordered),
                      'papers': sum(x['id'].startswith('P') for x in ordered),
                      'documents': sum(x['id'].startswith('D') for x in ordered)},
                     ensure_ascii=False))
    verified = []
    for item in ordered:
        source = Path(item['source'])
        match = hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
        verified.append({'id': item['id'], 'hash_matches_extraction': match})
    (WORK / '原件完整性检查.json').write_text(
        json.dumps(verified, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'unchanged_sources': sum(x['hash_matches_extraction'] for x in verified),
                      'checked': len(verified)}, ensure_ascii=False))
    for path in sorted(REVIEW.rglob('*.md')):
        if '_工作材料' in path.parts:
            continue
        text = path.read_text(encoding='utf-8')
        missing = []
        for target in re.findall(r'\]\(([^)]+)\)', text):
            target = target.strip('<>')
            if target.startswith(('http', '#')):
                continue
            target = target.split('#')[0]
            target = re.sub(r':\d+$', '', target)
            if not (path.parent / target).exists():
                missing.append(target)
        print(json.dumps({'note': str(path.relative_to(REVIEW)),
                          'chars': len(text), 'missing_links': missing},
                         ensure_ascii=False))

if __name__ == '__main__':
    main()
