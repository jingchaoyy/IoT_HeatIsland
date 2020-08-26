"""
Created on  1/14/20
@author: Jingchao Yang

Ground truth data correlation and alignment test
"""
import pandas as pd
import numpy as np

"""
Perform two approaches for estimation and inference of a Pearson
correlation coefficient in the presence of missing data: complete case
analysis and multiple imputation.
"""


def corr(X, Y):
    """Computes the Pearson correlation coefficient and a 95% confidence
    interval based on the data in X and Y."""

    r = np.corrcoef(X, Y)[0, 1]
    f = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(len(X) - 3)
    ucl = f + 2 * se
    lcl = f - 2 * se

    lcl = (np.exp(2 * lcl) - 1) / (np.exp(2 * lcl) + 1)
    ucl = (np.exp(2 * ucl) - 1) / (np.exp(2 * ucl) + 1)

    return r, lcl, ucl


## Read a data file with missing values
def convert_id(s):
    return float(s.split("-")[1])


Z = np.genfromtxt(r'E:\IoT_HeatIsland_Data\correlation_missing_data.csv', delimiter=",",
                  converters={0: convert_id}, skip_header=2,
                  missing_values=["NA", ])

## Complete case analysis

## The indices of cases with no missing values in columns 1 and 2
ii = np.flatnonzero(np.isfinite(Z[:, 1:3]).all(1))

## The correlation coefficients for complete case analysis
r, lcl, ucl = corr(Z[ii, 1], Z[ii, 2])

print("Complete case analysis:")
print("%.2f(%.2f,%.2f)" % (r, lcl, ucl))

## Use multiple imputation to estimate the correlation coefficient and
## standard error between columns 1 and 2.

## Columns of interest
X = Z[:, 1:3]

## Missing data patterns
ioo = np.flatnonzero(np.isfinite(X).all(1))
iom = np.flatnonzero(np.isfinite(X[:, 0]) & np.isnan(X[:, 1]))
imo = np.flatnonzero(np.isnan(X[:, 0]) & np.isfinite(X[:, 1]))
imm = np.flatnonzero(np.isnan(X).all(1))

## Complete data
XC = X[ioo, :]

## Number of multiple imputation iterations
nmi = 20

## Do the multiple imputation
F = np.zeros(nmi, dtype=np.float64)
for j in range(nmi):
    ## Bootstrap the complete data
    ii = np.random.randint(0, len(ioo), len(ioo))
    XB = XC[ii, :]

    ## Column-wise means
    X_mean = XB.mean(0)

    ## Column-wise standard deviations
    X_sd = XB.std(0)

    ## Correlation coefficient
    r = np.corrcoef(XB.T)[0, 1]

    ## The imputed data
    XI = X.copy()

    ## Impute the completely missing rows
    Q = np.random.normal(size=(X.shape[0], 2))
    Q[:, 1] = r * Q[:, 0] + np.sqrt(1 - r ** 2) * Q[:, 1]
    Q = Q * X_sd + X_mean
    XI[imm, :] = Q[imm, :]

    ## Impute the rows with missing first column
    ## using the conditional distribution
    va = X_sd[0] ** 2 - r ** 2 / X_sd[1] ** 2
    XI[imo, 0] = r * X[imo, 1] * (X_sd[0] / X_sd[1]) + \
                 np.sqrt(va) * np.random.normal(size=len(imo))

    ## Impute the rows with missing second column
    ## using the conditional distribution
    va = X_sd[1] ** 2 - r ** 2 / X_sd[0] ** 2
    XI[iom, 1] = r * X[iom, 0] * (X_sd[1] / X_sd[0]) + \
                 np.sqrt(va) * np.random.normal(size=len(iom))

    ## The correlation coefficient of the imputed data
    r = np.corrcoef(XI[:, 0], XI[:, 1])[0, 1]

    ## The Fisher-transformed correlation coefficient
    F[j] = 0.5 * np.log((1 + r) / (1 - r))

## Apply the combining rule, see, e.g.
## http://sites.stat.psu.edu/~jls/mifaq.html#howto
FM = F.mean()
RM = (np.exp(2 * FM) - 1) / (np.exp(2 * FM) + 1)
VA = (1 + 1 / float(nmi)) * F.var() + 1 / float(Z.shape[0] - 3)
SE = np.sqrt(VA)
LCL, UCL = FM - 2 * SE, FM + 2 * SE
LCL = (np.exp(2 * LCL) - 1) / (np.exp(2 * LCL) + 1)
UCL = (np.exp(2 * UCL) - 1) / (np.exp(2 * UCL) + 1)

print("\nMultiple imputation:")
print("%.2f(%.2f,%.2f)" % (RM, LCL, UCL))

# iot_path = r'E:\IoT_HeatIsland_Data\data\LA\exp_data\tempMatrix_LA.csv'
# wu_path = r'E:\IoT_HeatIsland_Data\data\LA\weather_underground\LA\processed_updated\KCALOSAN764.txt.csv'
#
# iot_data = pd.read_csv(iot_path, usecols=['time', '9q5csyd'])
# iot_data['time'] = pd.to_datetime(iot_data['time'], format='%Y%m%d%H')
# wu_data = pd.read_csv(wu_path, usecols=['time', 'temperature'])
#
# joined = wu_data.set_index('time').join(iot_data.set_index('time'))
# # joined.to_csv(r'E:\IoT_HeatIsland_Data\data\LA\exp_data\data_fusion\iot_wu_merge_updated.csv')
