# Security policy

## Supported versions

brotkit is pre-1.0. Only the latest released version gets security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/eoin-obrien/brotkit/security/advisories/new),
not as a public issue. Include the brotkit version, `brotkit doctor` output and a minimal
reproducer (for example, the file that triggers the problem).

You should get an acknowledgement within a week. Fixes are released as soon as practical and
credited in the changelog unless you'd rather stay anonymous.

## Safe-loading guarantee

Loading brotkit files must never execute code from the file. This is a design invariant, and
a violation is treated as a security vulnerability:

- **Formulas** are stored as a JSON expression tree restricted to whitelisted SymPy node types
  and a fixed set of elementary functions. User functions must be registered in code and
  appear in files by name only. brotkit never calls `eval`, `sympify`, `parse_expr` or loads
  `srepr` text from a file. Unknown nodes or names are refused with an error.
- **Arrays** load from NPZ with `allow_pickle=False`. Pickle is used only in-process (for
  example, between worker processes) and is never an archive format.
- **Result types** are resolved through a fixed registry keyed by name. No module or class
  named in a file is ever imported dynamically.

If you find a file that makes `brotkit.load`, `Formula.from_json` or any other loader run
code, read outside its input, or bypass these rules, please report it as above.
