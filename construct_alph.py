#**********************************************#
# This file contains procedures to dump single #
#   backbone on the disk as a PDB file.        #
#**********************************************#

import numpy as np
import Bio.PDB as pdb
# ...


def construct_pdb_from_alpha_mono(id: str,
                                  coord: np.ndarray, amino: list[str]):
    """Constructs and returns Bio.PDB.Structure of alpha-chain of monomer
       ready for dumping on disk."""
    builder = pdb.StructureBuilder.StructureBuilder()
    builder.init_structure(id)
    builder.init_model('GEN1')
    builder.init_chain('A')
    builder.init_seg(1)

    for i, amk in enumerate(amino):
        builder.init_residue(amk, ' ', i + 1, ' ')
        builder.init_atom('CA', list(coord[:, i]),
                          0.0, 1.0, ' ', ' CA ', element='C')

    return builder.get_structure()


def dump_struct(struct, id=None, out_format='prot_out/gen1_{0}.pdb') -> str:
    """Dumps structure (...)"""
    io = pdb.PDBIO()
    io.set_structure(struct)
    if id is None:
        id = struct.get_id()
    filename = out_format.format(id)
    io.save(filename)
    return filename


# if __name__ == '__main__':
#     import pickle as pkl
#     with open('./train.pkl', 'rb') as f:
#         train = pkl.load(f)
#         prot = train[21231]
#     struct = construct_pdb_from_alpha_mono(prot['id'],
#                                            prot['coord'],
#                                            prot['amino'])
#     dump_struct(struct)
