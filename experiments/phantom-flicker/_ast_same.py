"""One-off check that an edit changed only docstrings/comments: compares the AST of `git show
HEAD:<file>` with the working file, with every string-literal expression statement removed.

Run (in the repo): python _ast_same.py <repo> <path in repo>
"""

import ast
import subprocess
import sys


def strip(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [
                n for n in body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
            ] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


repo, path = sys.argv[1], sys.argv[2]
old = subprocess.run(["git", "-C", repo, "show", f"HEAD:{path}"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
new = open(f"{repo}/{path}", encoding="utf-8").read()
same = strip(ast.parse(old)) == strip(ast.parse(new))
print(f"{repo}/{path}: code identical to HEAD apart from docstrings/comments: {same}")
sys.exit(0 if same else 1)
