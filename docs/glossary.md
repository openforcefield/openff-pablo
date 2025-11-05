# Glossary

(glossary-page)=

:::{glossary}

graph
  A collection of nodes connected by edges.

chemical graph
  A {term}`graph` where atoms are nodes and bonds are edges.

molecular graph
  A {term}`connected` {term}`chemical graph` representing a single molecule.

complete chemical graph
complete molecular graph
  A {term}`chemical graph` or {term}`molecular graph` with all atoms and bonds present where all atoms have known elements and formal charges and all bonds have known bond orders. Contrast {term}`connectivity graph`.

connectivity graph
  A {term}`chemical graph` that includes all bonds and atoms, but not necessarily their formal charges or bond orders; ie, it defines the connectivity of a chemical system. Contrast {term}`molecular graph`, {term}`chemical graph`.

connected
  A connected graph is one in which for any two nodes $a$ and $b$ there exists some collection of edges that form a path from $a$ to $b$. That is, you can't split the graph into subgraphs that have no edges between them.

linking atom
  An atom in a residue or residue definition that forms a bond with an atom in another residue.

leaving atom
  An atom in a residue or residue definition that is removed when a bond between {term}`linking atom`s is formed.

:::
