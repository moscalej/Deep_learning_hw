from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from Code.SuperNet_v2 import model_S

# Hyperparams


weight_decay = Real(low=1e-6, high=1e-2, prior='log-uniform', name='weight_decay')
dim_drop = Real(low=1e-6, high=1e-2, prior='uniform', name='drop')
dim_factor = Real(low=0.5, high=0.8, prior='uniform', name='factor')
dim_patience = Integer(low=1, high=3, name='patience')
dim_s_1 = Integer(low=5, high=12, name='s_1')
dim_ex_1 = Integer(low=12, high=40, name='ex_1')
dim_s_2 = Integer(low=5, high=7, name='s_2')
dim_ex_2 = Integer(low=5, high=32, name='ex_2')
dim_epocs = Categorical(categories=[50], name='epochs')
dim_dense_num = Integer(low=1, high=5, name='dense_num')
dim_use_redux = Integer(low=0, high=1, name='use_redux')
dim_plus = Integer(low=0, high=1, name='plus')

## dimensions to explore:
dimensions = [
    weight_decay,
    dim_epocs,
    dim_factor,
    dim_patience,
    dim_s_1,
    dim_ex_1,
    dim_s_2,
    dim_ex_2,
    dim_use_redux,
    dim_drop,
    dim_plus,
    dim_dense_num,
]
defult = [1e-5, 30, 0.6, 2, 7, 10, 10, 20, 0,   0.3, 1, 62]
@use_named_args(dimensions=dimensions)
def uno(weight_decay=1e-5 ,epochs=30,factor=0.6 , patience=2, s_1=7,ex_1=10,s_2=10,ex_2=20,use_redux=False,
          drop = 0.3, plus= True, dense_num= 62):
    return model_S(weight_decay ,epochs,factor , patience, s_1,ex_1,s_2,ex_2,use_redux,
          drop, plus, dense_num)
## wrap function such that:
## Input:(one of each dimension defined)
## Output: accuracy per arch


## gp minimize
search_result = gp_minimize(func=uno,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=11,
                            # x0=defult
                            )
