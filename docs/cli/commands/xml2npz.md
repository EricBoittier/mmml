# `mmml xml2npz`

Molpro XML → NPZ.


## Usage

```bash
mmml xml2npz --help
```

## Options

```text
usage: mmml xml2npz [-h] -o OUTPUT [--padding PADDING] [--no-variables]
                    [--first-geometry] [--recursive] [--validate]
                    [--no-validate] [--strict] [--summary SUMMARY] [--quiet]
                    [--verbose] [--continue-on-error] [--max-files MAX_FILES]
                    inputs [inputs ...]

Convert Molpro XML files to standardized NPZ format

positional arguments:
  inputs                Input XML file(s) or directory/directories

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output NPZ file path
  --padding PADDING     Number of atoms to pad to (default: 60)
  --no-variables        Exclude Molpro internal variables from output
  --first-geometry      Use first geometry from files with multiple geometries
                        (default: use last/final)
  --recursive, -r       Recursively search directories for XML files
  --validate            Validate output NPZ file against schema
  --no-validate         Skip validation (faster but not recommended)
  --strict              Use strict validation (fail on warnings)
  --summary SUMMARY     Save conversion summary to JSON file
  --quiet, -q           Suppress progress output
  --verbose, -v         Verbose output
  --continue-on-error   Continue processing even if some files fail
  --max-files MAX_FILES
                        Maximum number of files to process (for testing)

Examples:
  # Convert single file
  mmml xml2npz output.xml -o data.npz
  
  # Convert multiple files
  mmml xml2npz file1.xml file2.xml file3.xml -o dataset.npz
  
  # Convert all XML files in directory
  mmml xml2npz molpro_outputs/ -o dataset.npz
  
  # Recursive search
  mmml xml2npz data/ -o dataset.npz --recursive
  
  # With validation and summary
  mmml xml2npz inputs/*.xml -o data.npz --validate --summary summary.json
  
  # Adjust padding for larger molecules
  mmml xml2npz inputs/*.xml -o data.npz --padding 100
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
