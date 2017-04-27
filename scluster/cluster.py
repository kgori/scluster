from scluster.evrot import cluster_rotate

class Result(object):

    def __init__(self, assignments, memberships, number, scores):
        self.assignments = assignments
        self.memberships = memberships
        self.number = number
        self.vectors = scores

    def __str__(self):
        return ("Found {} clusters:\n"
                "Cluster assignments:\n\t{}\n"
                "Cluster memberships:\n\t{}".format(self.number,
                                                    self.assignments,
                                                    [a.tolist() for a in self.memberships]))

def order(l):
    """
    This function preserves the group membership,
    but sorts the labelling into numerical order
    """
    from collections import defaultdict

    list_length = len(l)

    d = defaultdict(list)
    for (i, element) in enumerate(l):
        d[element].append(i)

    l2 = [None] * list_length

    for (name, index_list) in enumerate(sorted(d.values(), key=min),
                                        start=1):
        for index in index_list:
            l2[index] = name

    return tuple(l2)


def spectral_rotate(coordinates, min_groups=2, max_groups=None,
                    kmeans=False, verbose=True):
    (nclusters, clustering, quality_scores, rotated_vectors) = \
        cluster_rotate(coordinates, max_groups=max_groups,
                       min_groups=min_groups)

    translate_clustering = [None] * coordinates.shape[0]
    no_of_empty_clusters = 0
    for (group_number, group_membership) in enumerate(clustering):
        if len(group_membership) == 0:
            no_of_empty_clusters += 1
        for index in group_membership:
            translate_clustering[index - 1] = group_number
    T = order(translate_clustering)
    if no_of_empty_clusters > 0:
        print('Subtracting {0} empty {1}'.format(no_of_empty_clusters,
                                           ('cluster' if no_of_empty_clusters == 1 else 'clusters')))
        nclusters -= no_of_empty_clusters

    # ######################

    if verbose:
        print('Discovered {0} clusters'.format(nclusters))
        print('Quality scores: {0}'.format(quality_scores))
        # if kmeans:
        #     print('Pre-KMeans clustering: {0}'.format(clustering))
    # if kmeans:
    #     T = self.kmeans(nclusters, rotated_vectors)

    return Result(T, [cl for cl in clustering if len(cl > 0)], nclusters, quality_scores)
