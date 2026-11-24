"""Recover bounded file submissions from completed ACP read-back observations.

No historical commands are executed. Recovered text is not an original byte hash.
"""
import hashlib
import json
import re

from core.io import sha256

OUTPUTS = {
    'dapt-intrusion-detection': ['/root/network_stats.csv'],
    'lake-warming-attribution': ['/root/output/trend_result.csv', '/root/output/dominant_factor.csv'],
    'financial-modeling-qa': ['/root/answer.txt'],
}


def recover(attempt, source_manifest, task_id):
    if task_id not in OUTPUTS:
        raise ValueError('No certified trajectory text extractor for task')
    path = attempt / 'openhands/acp-events.json'
    expected = next((e['sha256'] for e in source_manifest if e['path'] == str(path)), None)
    if not expected or sha256(path) != expected:
        raise ValueError('Recovery requires an audited ACP source hash')
    events = json.loads(path.read_text())['events']

    def observation(needle):
        matches = []
        for row in events:
            e = row['event']
            if e.get('type') == 'tool_call' and needle in e.get('title', '') and e.get('status') == 'completed':
                body = '\n'.join(c['content']['text'] for c in e.get('content', []))
                matches.append((row, body))
        if not matches:
            raise ValueError('No completed, full file observation')
        return matches[-1]

    if task_id == 'dapt-intrusion-detection':
        row, text = observation('Reading /root/network_stats.csv')
        numbered = re.findall(r'^\s*(\d+)\t(.*)$', text, re.MULTILINE)
        if not numbered or [int(n) for n, _ in numbered] != list(range(1, len(numbered)+1)):
            raise ValueError('Non-contiguous or truncated numbered read-back')
        if numbered[0][1] != 'metric,value' or numbered[-1][1] != '':
            raise ValueError('Unexpected CSV read-back boundary')
        payloads = [('\r\n'.join(v for _, v in numbered[:-1])+'\r\n').encode()]
        method = 'Complete numbered CSV read-back; CRLF inferred from recorded csv.writer'
    elif task_id == 'lake-warming-attribution':
        row, text = observation('cat /root/output/trend_result.csv')
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
        if '✅ Exit code: 0' not in text:
            raise ValueError('CSV read-back did not complete successfully')
        trend, dominant = text.split('=== trend_result.csv ===\n', 1)[1].split('\n=== dominant_factor.csv ===\n')
        dominant = dominant.split('\n\n📁 Working directory:', 1)[0]
        if not trend.startswith('slope,') or not dominant.startswith('variable,contribution\n'):
            raise ValueError('Unexpected CSV boundaries')
        payloads = [trend.encode(), dominant.encode()]
        method = 'Complete successful cat output; Linux LF text'
    else:
        row, text = observation('Editing /root/answer.txt')
        args = json.JSONDecoder().raw_decode(row['event']['title'].split('file_editor: ', 1)[1])[0]
        if args.get('command') != 'create' or args.get('path') != '/root/answer.txt':
            raise ValueError('Expected literal answer file creation')
        _, confirmation = observation('cat /root/answer.txt')
        if '\x1b[?2004l'+args['file_text']+'\x1b[?2004h' not in confirmation or '✅ Exit code: 0' not in confirmation:
            raise ValueError('Answer create and read-back disagree')
        payloads = [args['file_text'].encode()]
        method = 'Literal create file_text corroborated by successful cat'
    if any(len(b)>65536 for b in payloads):
        raise ValueError('Recovery exceeds bounded text file size')
    return {'source_path': str(path), 'source_sha256': expected,
            'event_index': row['index'], 'tool_call_id': row['event']['tool_call_id'],
            'method': method, 'original_file_hash_available': False,
            'files': [{'source_path': p, 'hex': b.hex(), 'sha256': hashlib.sha256(b).hexdigest(),
                       'size': len(b), 'mode': '644', 'kind': 'file'}
                      for p, b in zip(OUTPUTS[task_id], payloads)]}
