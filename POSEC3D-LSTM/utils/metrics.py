import numpy as np

def hitAtK(result, k):
    if result.shape[1]>k:
        result = result[:,:k]
    count = 0
    for i in range(len(result)):
        hit = np.any(result[i,:])
        if hit:
            count += 1

    return count / len(result)

def precisionAtK(result, k):
    """
    Precision@K:
        Set a rank threshold K
        Compute % relevant in top K
        Ignores documents ranked lower than K
    """
    if k<result.shape[1]:
        result = result[:, :k]

    score = 0.0
    for i in range(len(result)):
        score += np.sum(result[i,:]) / k
    return score / len(result)

def AveragePrecision(result, k):
    """
        Consider rank position of each relevant doc
        K1,K2, … KR
        Compute Precision@K for each K1, K2, … KR
        Average precision = average of P@K
    """
    if len(result)>k:
        result = result[:k]
    score = 0.0
    num_hits = 1.0

    for i,p in enumerate(result):
        if p == 1.:
            score += num_hits / (i+1.0)
            num_hits += 1.0
    if score == 0.:
        return 0.
    return score / (num_hits-1)

def meanAveragePrecision(result, k):
    """
        Consider rank position of each relevant doc K1,K2, … KR
        Compute Precision@K for each K1, K2, … KR
        Average precision = average of P@K
    """
    apk = []
    for i in range(len(result)):
        apk.append(AveragePrecision(result[i,:], k))
    return np.mean(apk)