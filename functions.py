# CODE FROM JUAN: 
def neighbors(data, k=20):
    # for a given dataset, finds the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',n_jobs =-1).fit(data)
    distances, indices = nbrs.kneighbors()
    return indices

def jaccard(A,B):
    # for two sets A and B, finds the Jaccard distance J between A and B
    A = set(A)
    B = set(B)
    union = list(A|B)
    intersection = list(A & B)
    J = ((len(union) - len(intersection))/(len(union)))
    return(J)
def get_AJD(A,B, num_neigh=20):
    NA= neighbors(A,k=num_neigh)
    NB = neighbors(B,k=num_neigh)
    #num_neigh=NA.shape[1]
    total_JD=0
    for i in range(NA.shape[0]):
        total_JD+= jaccard(NA[i,:],NB[i,:])
    total_JD/= NA.shape[0]
    return total_JD

# ALSO SEE https://github.com/shamusc/ajd?tab=readme-ov-file