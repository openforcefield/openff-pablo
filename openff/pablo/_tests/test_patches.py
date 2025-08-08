from openff.pablo.ccd.patches import disambiguate_alt_ids
from openff.pablo.residue import (
    AtomDefinition,
    BondDefinition,
    ResidueDefinition,
    _skip_residue_definition_validation,
)


def test_disambiguate_alt_ids():
    with _skip_residue_definition_validation():
        with_synonyms = ResidueDefinition(
            atoms=(
                AtomDefinition.with_defaults(name="H1", symbol="H", synonyms=["H2"]),
                AtomDefinition.with_defaults(name="H2", symbol="H", synonyms=["H3"]),
                AtomDefinition.with_defaults(name="O", symbol="O", synonyms=["O1"]),
            ),
            bonds=(
                BondDefinition.with_defaults("O", "H1"),
                BondDefinition.with_defaults("O", "H2"),
            ),
            crosslink=BondDefinition.with_defaults("O", "H1", order=0),
            description="water with clashing synonyms",
            linking_bond=BondDefinition.with_defaults("O", "H2", order=0),
            residue_name="HOH",
            virtual_sites=(),
        )
        res1, res2 = disambiguate_alt_ids(with_synonyms)

    assert res1 == with_synonyms.replace(
        atoms=(
            AtomDefinition.with_defaults(name="H1", symbol="H"),
            AtomDefinition.with_defaults(name="H2", symbol="H"),
            AtomDefinition.with_defaults(name="O", symbol="O"),
        ),
    )

    assert res2.description == with_synonyms.description + " altids"
    assert set(res2.atoms) == {
        AtomDefinition.with_defaults(name="H2", symbol="H"),
        AtomDefinition.with_defaults(name="H3", symbol="H"),
        AtomDefinition.with_defaults(name="O1", symbol="O"),
    }
    assert set(res2.bonds) == {
        BondDefinition.with_defaults("O1", "H2"),
        BondDefinition.with_defaults("O1", "H3"),
    }
    assert res2.linking_bond == BondDefinition.with_defaults("O1", "H3", order=0)
    assert res2.crosslink == BondDefinition.with_defaults("O1", "H2", order=0)

    res1._validate()
    res2._validate()
