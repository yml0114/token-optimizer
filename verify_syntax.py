import ast
import sys
import os

os.chdir('/Users/liangliang/.qwenpaw/workspaces/default/token-optimizer')
files = [
    'src/token_optimizer/core/compression_store.py',
    'src/token_optimizer/core/prompt_reorderer.py',
    'src/token_optimizer/core/smart_compressor.py',
    'src/token_optimizer/production.py',
    'benchmark_headroom_integration.py',
]
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'SYNTAX ERROR in {f}: {e}')
        sys.exit(1)
print('All files pass syntax check.')
