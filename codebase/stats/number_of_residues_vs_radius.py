from matplotlib.pyplot import plot, show, xlabel, ylabel

from codebase.data.dbd4.dbd4 import DBD4
from codebase.data.dbd4.dbd4_feature_extractor import DBD4FeatureExtractor
from codebase.pairpred.model.enums import Features


__author__ = 'basir'


def main():
    seed = 1
    dbd4 = DBD4(size=5000, ratio=1, thresh=6, seed=seed)
    cum_neigh = []
    # radii = [8]
    radii = [8, 10, 15, 30, 45, 60, 75, 90, 100]
    for index, radius in enumerate(radii):
        DBD4FeatureExtractor.extract([Features.D2_PLAIN_SHAPE_DISTRIBUTION], dbd4,
                                     number_of_bins=20, radius=radius,
                                     number_of_samples=2000, seed=seed,
                                     gamma=0.5, save=True, folds=5, rasa=.5)
        res_no = 0
        cum_neigh.append(0)
        for protein_complex in dbd4.complexes.values():
            proteins = [protein_complex.unbound_formation.ligand, protein_complex.unbound_formation.receptor]
            for protein in proteins:
                for residue in protein.residues:
                    cum_neigh[index] += residue.get_feature(Features.NUMBER_OF_NEIGHBOURS)
                    res_no += 1
        cum_neigh[index] = cum_neigh[index]/res_no

    plot(radii, cum_neigh)
    xlabel("Radius "+r'$(\AA)$')
    ylabel("Average Number of Neighbouring Residues")
    show()


if __name__ == "__main__":
    main()