import argparse
import re
import sys


def anchor(node_name):
    """The HTML anchor for a texinfo node name.

    Menu entries and Node: lines spell the same node differently -- "* How to
    build a grid::" against "Node: How to build a grid" -- so both sides go
    through here and are guaranteed to agree.
    """
    return re.sub(r'\s+', '', node_name.strip())


def extract_title(info_lines):
    """Extract a title from the info file content."""
    for line in info_lines:
        line = line.strip()
        # Skip header, empty, navigation, and menu lines
        if not line or line.startswith('CHARMM Element'):
            continue
        if re.match(r'File:', line) or re.match(r'Up:', line):
            continue
        if re.match(r'\*\s+\S+::', line) or re.match(r'\*\s*Menu:', line):
            continue
        # First non-empty content line is the title
        if len(line) > 3:
            # Clean up whitespace
            return re.sub(r'\s+', ' ', line).strip()
    return 'CHARMM Documentation'


def to_html(info_lines, filename=''):
    """Convert CHARMM .info file lines to HTML."""
    title = extract_title(info_lines)

    # Escape a string for safe use in HTML attributes/text
    def esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    txt = []
    txt.append('<!DOCTYPE html>\n')
    txt.append('<html>\n')
    txt.append('<head>\n')
    txt.append('<meta charset="utf-8">\n')
    txt.append(f'<title>{esc(title)}</title>\n')
    txt.append('<link rel="stylesheet" href="charmmdoc.css">\n')
    txt.append('</head>\n')
    txt.append('<body>\n')
    txt.append('<div class="page-header">\n')
    txt.append(f'<h1>{esc(title)}</h1>\n')
    txt.append('<div class="nav">')
    txt.append('<a href="index.html">Index</a>')
    txt.append('<a href="charmm.html">CHARMM Home</a>')
    txt.append('<a href="commands.html">Commands</a>')
    txt.append('</div>\n')
    txt.append('</div>\n')
    txt.append('<pre>\n')

    section = None
    in_content = False

    # Does this file use texinfo nodes at all?  If not, the loop below would
    # otherwise emit nothing; see the fallback inside it.
    has_sections = any(re.match(r'File:', l.strip()) for l in info_lines)
    if not has_sections:
        print(f'{filename or "input"}: no "File: ... Node: ..." lines; '
              'emitting the text as one section with no table of contents. '
              'See doc/domdec.info for the expected layout.', file=sys.stderr)

    for line in info_lines:
        line = line.rstrip('\r\n')

        # Skip the CHARMM Element header line
        if line.startswith('CHARMM Element'):
            continue

        # Skip info control characters
        if re.match(r'^\x1F$', line):
            continue
        if re.match(r'^\S$', line) and len(line) == 1:
            continue

        # Parse File:/Node: lines (various separator styles: comma, -=-, spaces)
        #
        # The node name runs to a comma or to the next navigation keyword.
        # It is not restricted to letters: real node names include I/O,
        # C21-C22, Multi-GPU, 2DSVB and SQM_Syntax.  Matching only letters
        # here turned "Node: Variable-e14fac" into the anchor "Variable",
        # while the menu entry pointed at "#Variable-e14fac" -- so those
        # table-of-contents links led nowhere.
        # The space after "File:" and "Node:" is not always there either
        # ("File:Cons, Node:Syntactic Glossary"), which used to drop the node
        # on the floor and leave the section unreachable.
        node_match = re.search(
            r'File:\s*\S+\s*[,\-=\s]+\s*Node:\s*'
            r'(.+?)\s*(?:,|$|\s+(?:Up|Next|Previous|Prev):)', line)
        if node_match:
            node = anchor(node_match.group(1))

            # Close previous section
            if section is not None:
                txt.append('</div>\n')

            txt.append(f'<a id="{node}"></a>\n')
            txt.append('<div class="charmmdoc">\n')
            if section is not None:
                txt.append('<a href="#Top">Top</a>\n')

            section = node
            in_content = False
            continue

        # Skip standalone Up:/Previous:/Next: navigation lines
        if re.match(r'\s*Up:\s', line):
            continue

        # Skip Menu: header lines
        if re.search(r'\*\s*Menu:', line):
            continue

        # Only output content once we're inside a section.
        #
        # Unless the file has no sections at all.  The parser keys on
        # texinfo-style "File: ... Node: ..." markers, and a document written
        # without them used to come out as an empty <pre>, silently: the tool
        # exits 0 and writes a valid page with nothing in it.  Six shipped
        # that way for years.  Every document in doc/ now carries the markers,
        # so this is a net for the next one that does not: it warns above and
        # renders the text as written, which is worse than a navigable page
        # and far better than a blank one.
        if section is None and not has_sections:
            section = 'Top'
        if section is None:
            continue

        # Track whether we've seen any content
        if line.strip():
            in_content = True

        # Don't output blank lines before first content
        if not in_content:
            continue

        # Convert menu entries: * Name::  Description
        #
        # The node name may be several words ("How to build a grid") and may
        # itself contain a colon ("15N Chemical shift:Syntax" in ssnmr), so
        # the name is everything up to the first "::".  The anchor is that
        # name with its spaces removed -- the same spelling the Node: parser
        # above produces, so the two always agree.  This used to be three
        # regexes handling one, two and three words, which silently left any
        # longer name as plain text with no link.
        line = re.sub(
            r'\*\s+(\S.*?)::\s+(.*)',
            lambda m: (f'* <a href="#{anchor(m.group(1))}">{m.group(1)}</a>'
                       f' | {m.group(2)}'),
            line)

        # Convert menu entries that reference other docs:
        # * Name: (doc/file.info).  Description
        line = re.sub(
            r'\*\s+([\S ]+?)\s*:+\s*\(doc/(\S+?)\.info\s*\)\s*\.?\s*',
            r'* <a href="\2.html">\2</a>: ',
            line)

        # Convert *note cross-references: *note Name:(doc/file.info)Node
        def _note_with_node(m):
            name, doc, node = m.group(1), m.group(2), m.group(3)
            if node:
                return f'<a href="{doc}.html#{node}">{name}</a>'
            return f'<a href="{doc}.html">{name}</a>'
        line = re.sub(
            r'\*[nN]ote\s+([\S ]+?)\s*:\s*\(doc/(\S+?)\.info\)\s*([A-Za-z]*)',
            _note_with_node,
            line)

        # Convert remaining *note references without doc/ prefix
        line = re.sub(
            r'\*[nN]ote\s+([\S ]+?)\s*:\s*\((\S+?)\.info\)',
            r'<a href="\2.html">\1</a>',
            line)

        # Convert bare (doc/file.info) references
        line = re.sub(
            r'\(doc/(\S+?)\.info\)',
            r'(<a href="\1.html">\1</a>)',
            line)

        # Convert "see file.info" references
        line = re.sub(
            r'[sS]ee\s+(\S+)\.info',
            r'see <a href="\1.html">\1</a>',
            line)

        # Convert "Prerequisite reading: file.info"
        line = re.sub(
            r'(Prerequisite reading:)\s+(\S+)\.info',
            r'\1 <a href="\2.html">\2</a>',
            line)

        # Strip * Info: (self-referential info reader help)
        line = re.sub(r'\*\s+Info:\s+\(Info\).*', '', line)

        # Escape HTML special chars in text, preserving tags we inserted
        # Split on HTML tags, escape only non-tag parts
        parts = re.split(r'(<a\s[^>]*>|</a>)', line)
        for i, part in enumerate(parts):
            if not part.startswith('<'):
                part = part.replace('&', '&amp;')
                part = part.replace('<', '&lt;')
                part = part.replace('>', '&gt;')
                parts[i] = part
        line = ''.join(parts)

        txt.append(line + '\n')

    # Close final section
    if section is not None:
        txt.append('</div>\n')

    txt.append('</pre>\n')
    txt.append('</body>\n')
    txt.append('</html>\n')
    return txt


def main():
    parser = argparse.ArgumentParser(
        prog='info2html.py',
        description='converts CHARMM .info documentation files to HTML')
    parser.add_argument('info_file_name')
    parser.add_argument('html_file_name')
    args = parser.parse_args()

    # UTF-8, not ASCII: contributors' names are not all spellable in ASCII
    # (the MLpot author is Toepfer with an o-umlaut), and neither are the
    # units and symbols chemistry documentation wants -- Angstrom, degree,
    # plus-minus, Greek.  Reading as ASCII made those a hard error here, so
    # the only way to accept such a file was to transliterate it first.
    with open(args.info_file_name, 'r', encoding='utf-8') as info_file:
        info_lines = info_file.readlines()

    html = to_html(info_lines, args.info_file_name)

    # Name the encoding on the way out too.  Without it Python uses whatever
    # the machine's locale happens to be, so the same .info file produced
    # different bytes on different machines -- and the page said it was ASCII
    # either way, which is not a charset label a browser recognises.
    with open(args.html_file_name, 'w', encoding='utf-8') as html_file:
        html_file.writelines(html)


if __name__ == "__main__":
    main()
