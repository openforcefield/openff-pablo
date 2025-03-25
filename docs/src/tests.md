# PDB Test Suite

Pablo has a (developing) PDB test suite.

<!-- :::{list-table}
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
*   - 2zuq_prepared.pdb
    - disulfide bond between two different protein chains, with additional chains between and after the crosslinked chains
*   - 2MUM_neutralized.pdb
    - all 20 canonical amino acids, neutral terminii, neutral canonical AA side chains
*   - 1p3q_noter.pdb
    - missing ter record between protein chains with same chain id but incompatible terminii, missing ter record between protein chains with different chain id
*   - 1hje_samechain.pdb
    - Disulfide bonds within chain, missing ter record between polymer and non-polymer chains with same chain id, NH2 cap, strained conformation
*   - 1hje_diffchain.pdb
    - Disulfide bonds within chain, missing ter record between polymer and non-polymer chains with same different id, NH2 cap, strained conformation
*   - 1A4T.pdb
    - NMR model direct from PDB, multi-model file, RNA and protein in same file
*   - 1a45_samechain.pdb
    - RNA and protein in same file, missing TER record between incompatible polymers with same chain ID
*   - 2MUM_blowup.pdb
    - Strained conformation
*   - 5eil_fixed.pdb
    - Non-canonical amino acid from CCD, iron ion
*   - 3ip9_dye_solvated.pdb
    - Large system with water and NaCl, non-canonical amino acid not in CCD
::: -->

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
