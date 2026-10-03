import argparse
import os


def create_header():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CHARMM Documentation Index</title>
<link rel="stylesheet" href="charmmdoc.css">
</head>
<body>
<div class="page-header">
<h1>CHARMM Documentation Index</h1>
<div class="nav">
<a href="charmm.html">CHARMM Home</a>
<a href="commands.html">Commands</a>
</div>
</div>
<div class="index-body">
"""


def create_footer():
    return """</div>
</body>
</html>
"""


def create_pycharmm_index():
    return """<h2>pyCHARMM Documentation</h2>
<ul>
<li><a href="pycharmm.html">About pyCHARMM</a></li>
<li><a href="pycharmm_index.html">pyCHARMM Python API</a></li>
</ul>
"""


def create_charmm_script_index(files):
    from collections import OrderedDict

    # Group files by first letter
    groups = OrderedDict()
    for filename in sorted(files):
        name = filename[:-5]  # strip .html
        letter = name[0].upper()
        if letter not in groups:
            groups[letter] = []
        groups[letter].append((filename, name))

    # Letter navigation bar
    html = '<h2>CHARMM Documentation</h2>\n'
    html += '<div class="letter-nav">\n'
    for letter in groups:
        html += f'<a href="#letter-{letter}">{letter}</a>\n'
    html += '</div>\n'

    # Sections by letter
    for letter, entries in groups.items():
        html += f'<a id="letter-{letter}"></a>\n'
        html += f'<h3>{letter}</h3>\n'
        html += '<ul>\n'
        for filename, name in entries:
            html += f'<li><a href="{filename}">{name}</a></li>\n'
        html += '</ul>\n'

    return html


def main():
    parser = argparse.ArgumentParser(
        prog='index.py',
        description='creates index based on directory listing')
    parser.add_argument('dir_to_index',
                        help='directory of info files to index')
    parser.add_argument('path_to_index',
                        help='directory where index.html will be written')
    parser.add_argument('-p', '--pycharmm', action='store_true',
                        help='link to pycharmm docu in index output')
    args = parser.parse_args()

    html = create_header()
    if args.pycharmm:
        html += create_pycharmm_index()

    # Get the list of files (not directories) in the directory
    files = [os.path.basename(f) for f in os.listdir(args.dir_to_index)
             if (os.path.isfile(os.path.join(args.dir_to_index, f))
                 and f.endswith('.info'))]

    files = [f[:-5] + '.html' for f in files]

    html += create_charmm_script_index(files)
    html += create_footer()

    index_path = os.path.join(args.path_to_index, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html)


if __name__ == "__main__":
    main()
