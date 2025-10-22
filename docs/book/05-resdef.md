(chapter:resdef)=
# Creating custom residue definitions

Sometimes you need to create a custom residue definition. This might be when you want to augment the CCD cache with a residue of your own, or when creating your own residue library. All residue definitions are instances of the [`ResidueDefinition`] class. That class has a large number of `from_*()` methods that can be used to easily and quickly generate residue definitions.

For ligands and single-residue molecules, the reference docs for that class might be all you need. Unfortunately, preparing residue definitions that can actually link to each other in a polymer requires a little bit more understanding.

This chapter will first go over the basics of how a `ResidueDefinition` encodes what information, and then will summarize the important ways to create one.

## What's in a residue definition

A typical residue

### Links and crosslinks

### Anonymous definitions

## Creating a named residue definition

### From a SMILES string

### From an OpenFF Molecule

## Creating an anonymous residue definition

### From a SMILES string

### From an OpenFF Molecule

### From an SDF file

[`ResidueDefinition`]: openff.pablo.ResidueDefinition
