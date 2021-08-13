import torch

def getUncertainty(V, classNum):
    # For classification, input V is the possibility vector for each category
    # Get uncertainty from OUI (task model)
    # Only compute uncertainty of unlabeled data

    # V: dataNum * classNum
    maxV = torch.amax(V, axis=1)       # max of each category
    varV = torch.var(V, axis=1, unbiased=False)        # variance of each data point

    C = classNum
    C_inv = 1 / C

    temp1 = torch.pow(C_inv - maxV,2)
    temp2 = (C - 1) * torch.pow(C_inv - (1 - maxV) / (C - 1), 2)     ##
    minVarV = C_inv * (temp1 + temp2)

    uncertainty_temp1 = torch.divide(minVarV, varV)
    uncertainty_temp2 = torch.multiply(uncertainty_temp1, maxV)
    uncertainty = 1 - uncertainty_temp2

    return uncertainty  # return the overall uncertainty of every unlabeled data point
