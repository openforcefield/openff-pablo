# PDB Test Suite

Pablo has a (developing) PDB test suite.

:::{list-table}
:header-rows: 1
:name: pdb-tests
:align: left
:width: 100%
:widths: 25 75

*   - File name <input type="text" id="pdb-filter-0" onkeyup="filterTable(0)" placeholder="Filter..." style="display: block; width: 8rem">
    - Coverage <input type="text" id="pdb-filter-1" onkeyup="filterTable(1)" placeholder="Filter..." style="display: block; width: 8rem">
*   - 5ap1_prepared.pdb
    - ACE/NME caps, `unique_molecules`
*   - 193L_prepared.pdb
    - disulfides, all 20 canonical amino acids, charged terminii, charged canonical AA side chains
*   - 2hi7_prepared.pdb
    - disulfide bond between two different protein chains
*   - 1p3q_noter.pdb
    - missing ter record between protein chains with same chain id but incompatible terminii, missing ter record between protein chains with different chain id
*   - 1hje_samechain.pdb
    - Disulfide bonds within chain, missing ter record between polymer and non-polymer chains with same chain id, NH2 cap, strained conformation
*   - 1hje_diffchain.pdb
    - Disulfide bonds within chain, missing ter record between polymer and non-polymer chains with same different id, NH2 cap, strained conformation
*   - 1A4T.pdb/.cif
    - NMR model direct from PDB, multi-model file, RNA and protein in same file
*   - 1a45_samechain.pdb
    - RNA and protein in same file, missing TER record between incompatible polymers with same chain ID
*   - 5eil_fixed.pdb
    - Non-canonical amino acid from CCD, iron ion
*   - 3ip9_dye_solvated.pdb
    - Large system with water and NaCl, non-canonical amino acid not in CCD
*   - big_bilayer.pdb
    - Very large system, 4 character residue names, hexadecimal atom and residue numbers
*   - capped_arg_altresonance.pdb
    - Formal charge on incorrect resonance-equivalent atom in arginine
*   - capped_arg_altresonance_conect.pdb
    - Formal charge on incorrect resonance-equivalent atom in arginine with CONECT records
*   - capped_ser_extrater
    - erroneous mid-chain TER record (should raise error)
*   - e2_7nel
    - `unique_molecules`
*   - microviridin_edited.pdb
    - Extensive nonstandard cross-links with CONECT records and custom residue names
*   - prepared_pdbs/2MUM_neutralized.pdb
    - all 20 canonical amino acids, neutral terminii, neutral canonical AA side chains
*   - prepared_pdbs/2MUM_blowup.pdb
    - Strained conformation, all 20 canonical amino acids, neutral terminii, neutral canonical AA side chains
*   - prepared_pdbs/2MUM_composed_function.pdb
    - discontiguous residue numbers, discontiguous atom serials, insertion codes, hexadecimal residue numbers, hexadecimal atom serials, re-used residue numbers, re-used atom serials, all 20 canonical amino acids
*   - prepared_pdbs/2MUM_icode.pdb
    - insertion codes, all 20 canonical amino acids
*   - prepared_pdbs/2MUM_letters_in_resseq.pdb
    - hexadecimal residue numbers, all 20 canonical amino acids
*   - prepared_pdbs/2MUM_letters_in_serial.pdb
    - hexadecimal atom serials, all 20 canonical amino acids
*   - prepared_pdbs/2MUM_reuse_resseq.pdb
    - re-used residue numbers, all 20 canonical amino acids
*   - prepared_pdbs/2MUM_reuse_serial.pdb
    - re-used atom serials, all 20 canonical amino acids
*   - prepared_pdbs/2zuq_prepared.pdb
    - disulfide bond between two different protein chains, with additional chains between and after the crosslinked chains
*   - prepared_pdbs/3h34_prepared.pdb
    - correctly named ligand in CCD with CONECT records
*   - prepared_pdbs/ions.pdb
    - All Sage monatomic ions (correctly named)
*   - prepared_pdbs/polyglycines.pdb
    - Stress-test chain termination
:::

<script>
  function filterTable(column) {
    var input, filter, table, tr, td, i, txtValue;
    input = document.getElementById(`pdb-filter-${column}`);
    filter = input.value.toUpperCase();
    table = document.getElementById("pdb-tests");
    tr = table.getElementsByTagName("tr");
    for (i = 0; i < tr.length; i++) {
      td = tr[i].getElementsByTagName("td")[column];
      if (td) {
        txtValue = td.textContent || td.innerText;
        console.log(column, txtValue)
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
          tr[i].style.display = "";
        } else {
          tr[i].style.display = "none";
        }
      }
    }
  }
</script>
