"""Verify schema-independent checks still reject real source changes; no Torch needed."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch
import check_structure

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert check_structure.check_preserved_methods(ROOT) == 16
    native_parse = ast.parse
    def pre312_parse(*args, **kwargs):
        tree = native_parse(*args, **kwargs)
        for node in ast.walk(tree):
            if 'type_params' in node._fields:
                node._fields = tuple(field for field in node._fields if field != 'type_params')
        return tree
    with patch.object(check_structure.ast, 'parse', side_effect=pre312_parse):
        assert check_structure.check_preserved_methods(ROOT) == 16
    def altered_parse(*args, **kwargs):
        tree = native_parse(*args, **kwargs)
        if kwargs.get('filename', '').endswith('head.py'):
            cls = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.ClassDef) and node.name == '_HeadCore')
            method = next(node for node in cls.body
                          if isinstance(node, ast.FunctionDef) and node.name == '__init__')
            method.body.append(ast.Pass())
        return tree
    with patch.object(check_structure.ast, 'parse', side_effect=altered_parse):
        try:
            check_structure.check_preserved_methods(ROOT)
        except AssertionError as error:
            assert 'preserved implementation changed' in str(error)
        else:
            raise AssertionError('Changed model source was accepted')
    metadata = copy.deepcopy(json.loads((ROOT / 'PROVENANCE.json').read_text()))
    metadata['source_sha256'] = metadata['normalized_source_sha256'] = '0' * 64
    try:
        check_structure.check_preserved_methods(ROOT, metadata)
    except AssertionError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Untrusted archived source was accepted')
    print('PASS: native/pre-3.12 schema comparison; altered methods and archive checksums rejected')


if __name__ == '__main__':
    main()
