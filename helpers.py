"""
Code for some helper dictionaries.
"""
from distributions import sample_weibull, WeibullDensity
from models import Normal_LDA_Density, Normal_QDA_Density

DISTRIBUTION_TO_DENSITY = {
    "lda": Normal_LDA_Density, 
    "qda": Normal_QDA_Density, 
    "weibull": WeibullDensity
}

DISTRIBUTION_TO_SAMPLE_GENERATION = {
    "weibull": sample_weibull
}