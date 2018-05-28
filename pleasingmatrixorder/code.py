import numpy as np
import scipy.cluster
import scipy as sp
import scipy.optimize
import sys

def loss(confusion,clusters):
    L=0
    for i in range(len(clusters)):
        inside=np.sum(confusion[np.ix_(clusters[i][0],clusters[i][1])])
        outside = np.sum(confusion[clusters[i][0]]) + np.sum(confusion[:,clusters[i][1]]) - inside
        L -= np.log(inside/outside)
    return L

def add_new_singleton(confusion,clusters,idx,axis):
    Ls=np.zeros(len(clusters))
    for i in range(len(clusters)):
        clusters2=list(clusters)
        clusters2[i]=list(clusters2[i][0]),list(clusters2[i][1])
        clusters2[i][axis].append(idx)
        Ls[i]=loss(confusion,clusters2)
    
    best=np.argmin(Ls)
    clusters2=list(clusters)
    clusters2[best]=list(clusters2[best][0]),list(clusters2[best][1])
    clusters2[best][axis].append(idx)
    return clusters2
        

def perform_merge(confusion,clusters):
    Ls=np.ones((len(clusters),len(clusters)))*np.inf
    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            clusters3=list(clusters)
            AA=clusters3.pop(j)
            BB=clusters3.pop(i)
            clusters3.append((AA[0]+BB[0],AA[1]+BB[1]))
            Ls[i,j]=loss(confusion,clusters3)

    i,j = np.unravel_index(np.argmin(Ls),Ls.shape)

    clusters3=list(clusters)
    AA=clusters3.pop(j)
    BB=clusters3.pop(i)
    clusters3.append((AA[0]+BB[0],AA[1]+BB[1]))
    Ls[i,j]=loss(confusion,clusters3)

    return clusters3,Ls[i,j],i,j


def pnn(x,space=True): 
    sys.stdout.write(str(x) + (" " if space else ''))
    sys.stdout.flush()

def find_pleasing_order(confusion,mergify=True):

    ####################################3
    # get initial hungarian 

    # hungarian algorithm
    posA,posB=sp.optimize.linear_sum_assignment(-confusion)

    # groups which are null are deleted
    goodguys=np.diag(confusion[np.ix_(posA,posB)])>0
    posA=posA[goodguys]
    posB=posB[goodguys]

    clusters=[([x],[y]) for (x,y) in zip(posA,posB)]

    ################################3
    # get anybody we left out

    missing=set(list(range(confusion.shape[1]))).difference(set(posB))
    for m in missing:
        clusters=add_new_singleton(confusion,clusters,m,1)
        
    missing=set(list(range(confusion.shape[0]))).difference(set(posA))
    for m in missing:
        clusters=add_new_singleton(confusion,clusters,m,0)

    ################################
    # perform the merges

    if mergify:
        for k in range(len(clusters)-1):
            pnn("%d/%d"%(k,len(clusters)-1))
            clusters,L,i,j=perform_merge(confusion,clusters)

        return clusters[0]
    else:
        posA=[]
        posB=[]
        for aa,bb in clusters:
            posA.extend(aa)
            posB.extend(bb)

        return (posA,posB)