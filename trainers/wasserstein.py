import torch 

# calculate the 1 wasserstein distance 
def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)

def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
        Compute, between two one-dimensional distributions :math:`u` and
        :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
        statistical distance that is defined as:
        .. math::
            l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
        p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
        gives the energy distance.
        Parameters
        ----------
        u_values, v_values : tensors
            Values observed in the (empirical) distribution.
        u_weights, v_weights : tensors, optional
            Weight for each value. If unspecified, each value is assigned the same
            weight.
            `u_weights` (resp. `v_weights`) must have the same length as
            `u_values` (resp. `v_values`). If the weight sum differs from 1, it
            must still be positive and finite so that the weights can be normalized
            to sum to 1.
        Returns
        -------
        distance : float
            The computed distance between the distributions.
        Notes
        -----
        The input distributions can be empirical, therefore coming from samples
        whose values are effectively inputs of the function, or they can be seen as
        generalized functions, in which case they are weighted sums of Dirac delta
        functions located at the specified values.
        References
        ----------
        .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
            Munos "The Cramer Distance as a Solution to Biased Wasserstein
            Gradients" (2017). :arXiv:`1705.10743`.
        """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)
    # default concatenate on axis 0 
    all_values = torch.cat((u_values, v_values))
    # there is only one dimension, so it should work same in all cases
    all_values = torch.sort(all_values, stable=True).values
    

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values)
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], side="right").double()
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], side="right").double()
   
    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size(dim=0)
    else:
        u_sorted_cumweights = torch.cat((torch.tensor[0],
                                            torch.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size(dim=0)
    else:
        v_sorted_cumweights = torch.cat((torch.tensor[0],
                                            torch.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]
    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    # u_cdf.retain_grad()
    # 1 wasserstein loss
    if p == 1:
        return torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas))
    # 2 wasserstein loss 
    if p == 2:
        return torch.sqrt(torch.sum(torch.multiply(torch.square(u_cdf - v_cdf), deltas)))
    return torch.power(torch.sum(torch.multiply(torch.power(torch.abs(u_cdf - v_cdf), p),
                                    deltas)), 1/p)

def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as tensors.
    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.
    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.
    """
    # Validate the value array.
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")
    # print(values.is_leaf)
    # Validate the weight array, if specified.
    if weights is not None:
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if torch.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < torch.sum(weights) < float("Inf"):
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None
    

def main():
    a, b = torch.tensor([15, 4, 2], dtype=torch.float32, requires_grad=True), torch.tensor([5, 6, 5], dtype=torch.float32)
    loss = wasserstein_distance(a, b)
    print(loss)
    loss.backward()
    print(a.grad)    


if __name__ == "__main__":
    main()
