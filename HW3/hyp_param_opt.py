from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args


# Hyperparams
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_decay_rate = Real(low=1e-4, high=1e-6, prior='log-uniform', name='decay_rate')
dim_num_conv_Bulks = Integer(low=1, high=5, name='num_conv_Bulks')
dim_kernel_size = Categorical(categories=[3, 5], name='kernel_size')
dim_activation = Categorical(categories=['sigmoid', 'linear', 'relu'], name='activation')

## dimensions to explore:
dimensions = [dim_learning_rate, dim_num_conv_Bulks, dim_kernel_size, dim_activation]

## wrap function such that:
## Input:(one of each dimension defined)
## Output: accuracy per arch


## gp minimize
search_result = gp_minimize(func=train_wrapper,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=11,
                            x0=default_parameters)
