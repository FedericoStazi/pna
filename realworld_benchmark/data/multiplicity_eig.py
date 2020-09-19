from molecules import MoleculeDataset
from superpixels import SuperPixDataset
from SBMs import SBMsDataset
from COLLAB import COLLABDataset
from scipy import sparse as sp


def get_eig_val(g, pos_enc_dim=7, norm='none'):
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    if norm == 'none':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1), dtype=float)
        L = N * sp.eye(g.number_of_nodes()) - A
    elif norm == 'sym':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N
    elif norm == 'walk':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1., dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=2e-1)
    return EigVal


def get_multiplicity(DATASET_NAME, tol, dim, norm):
    if DATASET_NAME == 'ZINC':
        dataset = MoleculeDataset(DATASET_NAME)
    elif DATASET_NAME == 'SBM_PATTERN':
        dataset = SBMsDataset(DATASET_NAME)
    elif DATASET_NAME == 'CIFAR10':
        dataset = SuperPixDataset(DATASET_NAME)
    elif DATASET_NAME == 'COLLAB':
        dataset = COLLABDataset(DATASET_NAME)

    if DATASET_NAME == 'COLLAB':
        pass
    else:
        train_graphs = dataset.train.graph_lists
        val_graphs = dataset.val.graph_lists
        test_graphs = dataset.test.graph_lists
        train_eigs = [get_eig_val(g, pos_enc_dim=dim, norm=norm) for g in train_graphs]
        val_eigs = [get_eig_val(g, pos_enc_dim=dim, norm=norm) for g in val_graphs]
        test_eigs = [get_eig_val(g, pos_enc_dim=dim, norm=norm) for g in test_graphs]
        eigs = train_eigs + val_eigs + test_eigs
        i = 0
        n = len(eigs)
        for eig in eigs:
            if abs(eig[1] - eig[2]) < tol:
                i += 1
        return i / n


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--DATASET_NAME', help='Name of the Dataset')
    parser.add_argument('--tol', default=1e-3, help='Tolerance for multiplicity')
    parser.add_argument('--lap_norm', default='none', help='Normalisation for the Laplacian matrix')
    parser.add_argument('--dim', help='Number of eigs to compute')
    args = parser.parse_args()

    print(get_multiplicity(args.DATASET_NAME, args.tol, args.dim, args.lap_norm))

