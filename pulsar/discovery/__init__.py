"""
this module uses other modules such stattests, techniques and visualization
it should provide users some pre-composed or parametrized pipelines
to which they can feed the data and get constructed reports with
    - inference results
    - data summary
    - supporting visualization
"""
from pulsar.discovery.distributions import (
    EmpiricalDistribution,
    bucketize,
    custom_bucketize,
    check_distr,
    fit_distribution,
    is_binary_feature
)
