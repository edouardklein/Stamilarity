'''The stamilarity module provide tools to test the STAtistical siMILARITY of
two samples, i.e. to test wether both samples were generated from the same
distribution.

.. autofunction:: similar
'''

from scipy.stats import binom_test
from scipy.stats import chisquare
from collections import Counter, defaultdict
import numpy as np


def divide_in_groups(sample, bins=10, min_card=5):
    '''Return a list of booleans arrays suitable for indexing another array
    with the same dimensions as sample, based on groups devised from the
    values in sample.

    Sample should contain numbers (not strings) and no NaNs.
    '''
    unique_values = set(sample)
    assert len(unique_values) > 1, "There is only one group"
    bins = min(len(unique_values), bins)
    histo, edges = np.histogram(sample, bins=bins)
    indices = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high != edges[-1]:
            indices.append((sample >= low) & (sample < high))
        else:
            indices.append((sample >= low) & (sample <= high))
    while any(np.array(list(map(sum, indices))) <= min_card):
        assert len(indices) > 2, """Groups are too small, even when grouping the
        smallest ones together"""
        two_smallest_bins = sorted(range(len(indices)),
                                   key=lambda i: sum(indices[i]))[:2]
        others = [indices[i] for i in range(len(indices))
                  if i not in two_smallest_bins]
        mashed_up = indices[two_smallest_bins[0]] | indices[two_smallest_bins[1]]
        indices = [mashed_up] + others
    return indices


def empirical_discrete_distrib(*samples):
    '''Return, as a dict, the empirical distribution from the aggregate of the
    given samples of a discrete random variable.'''
    aggreg = sum(map(list, samples), [])
    cnt = Counter(aggreg)
    answer = defaultdict(lambda: 0)
    answer.update({k: cnt[k]/len(aggreg) for k in cnt})
    return answer


def similar(*args, distrib=None, continuous=False):
    '''Return the p-value of the hypothesis that all samples were sampled from
    the same distribution.

    Unless distrib is given, we use the union of all the samples as the
    theoretical distribution in our test's hypothesis.

    If continuous is True,

    ..todo::
    Write code that handles continuous distributions

    If continuous is False, samples are treated as drawn from a categorical
    random variable.

    If the samples appear to be drawn from a binary set, we use a binomial
    test. Otherwise we run a \Xi^2.

    Examples
    --------
    >>> import random
    >>> import stamilarity

    Compare two samples from the same binary distribution :

    >>> fair_coin1 = [1 if random.random()>.5 else 0 for i in range(10000)]
    >>> random.seed(123)
    >>> fair_coin2 = [1 if random.random()>.5 else 0 for i in range(10000)]
    >>> p = stamilarity.similar(fair_coin1, fair_coin2)
    >>> p > .05  # Fair coins
    True

    Compare two samples from the same categorical distribution :

    >>> fair_dice1 = [random.choice(range(6)) for i in range(10000)]
    >>> random.seed(456)
    >>> fair_dice2 = [random.choice(range(6)) for i in range(10000)]
    >>> p = stamilarity.similar(fair_dice1, fair_dice2)
    >>> p > .05  # Fair dices
    True

    Compare a sample against a theoretical binary distribution:

    >>> random.seed(789)
    >>> biased_coin = [1 if random.random()>.6 else 0 for i in range(10000)]
    >>> p = stamilarity.similar(biased_coin, distrib={1: .4, 0: .6})
    >>> p > .05
    True

    Compare a sample agains a theoretical categorical distribution:

    >>> biased_dice = [random.choice(range(6)) if random.random()>.1 else 0\
                       for i in range(10000)]
    >>> p = stamilarity.similar(biased_dice, {0: .16,\
                                              1: .15,\
                                              2: .15,\
                                              3: .15,\
                                              4: .15,\
                                              5: .15})
    >>> p > .05
    True

    Compare two dissimilar binary samples

    >>> p = stamilarity.similar(fair_coin1, biased_coin)
    >>> p > .05  # Detect a biased coin
    False

    Compare two dissimilar categorical samples:

    >>> p = stamilarity.similar(fair_dice1, biased_dice)
    >>> p > .05  # Detect a unfair dice
    False


    Parameters
    ----------
    args : iterables
        The experimental samples
    distrib: dict or None
        The theoretical distribution, as a dict. If None, the empirical
        distribution will be computed from the union of the iterables in args.
    continuous: bool
        If true, use a statistical tool that works with continuous
        distributions.

    Return
    ------
    p: float
        Return the p-value of the hypothesis that the sample are drawn from the
        same distribution.

    References
    ----------
    P-values are very often a misunderstood concept. Please make sure you
    know how to interpret the results. A good starting point is :
    http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2895822/
    '''
    assert len(args) >= 2 or len(args) == 1 and distrib, """Strictly more than
    one sample is required"""
    if not distrib and not continuous:
        distrib = empirical_discrete_distrib(*args)
    elif not distrib and continuous:
        raise Exception('Not implemented yet')
    sampled_from_distrib = lambda sample: sampled_from(sample, distrib,
                                                       continuous)
    return min(map(sampled_from_distrib, args))


def sampled_from(sample, distrib, continuous=False):
    '''Return the p-value of the hypothesis that sample was sampled from
    distrib.'''
    assert continuous or set(sample) <= set(distrib.keys())
    if not continuous and len(distrib.keys()) == 2:
        return run_binomial_test(sample, distrib)
    elif not continuous and len(set(distrib)) > 2:
        return run_chi_squared(sample, distrib)
    else:  # continuous
        raise Exception('Not implemented yet')


def run_binomial_test(sample, distrib, assymmetry_threshold=0.01):
    '''Run the binomial test and return the p-value of the hypothesis that
    sample was sampled from empirical distridution distrib'''
    zero = list(distrib.keys())[0]
    one = list(distrib.keys())[1]
    assert distrib[zero] > assymmetry_threshold and distrib[one] > assymmetry_threshold, """User
    is testing for similarity to an assymmetric distribution without setting a
    proper assymmetry threshold"""
    sample_1 = (np.array(sample) == one).sum()
    sample_0 = len(sample) - sample_1
    return binom_test([sample_1, sample_0], p=distrib[one])


def run_chi_squared(sample, distrib):
    '''Run the chi-squared test and return the p-value of the hypothesis that
    sample was sampled from empirical distribution distrib'''
    actual = Counter(sample)
    expected = {k: distrib[k]*len(sample) for k in distrib}
    _, answer = chisquare([actual[k] for k in sorted(distrib.keys())],
                          [expected[k] for k in sorted(distrib.keys())])
    return answer
