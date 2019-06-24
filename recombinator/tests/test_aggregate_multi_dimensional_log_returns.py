import numpy as np

from recombinator.log_returns import aggregate_multi_dimensional_log_returns

example_array = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]],
                          [[100, 200], [300, 400], [500, 600], [700, 800]]])

expected_aggregated_array \
    = np.array([[[4, 6], [12, 14]], [[400, 600], [1200, 1400]]])


def test_aggregate_multi_dimensional_log_returns():
    aggregated_example \
        = aggregate_multi_dimensional_log_returns(
            log_returns=example_array,
            number_of_observations_to_aggregate=2)

    assert np.allclose(aggregated_example, expected_aggregated_array)
