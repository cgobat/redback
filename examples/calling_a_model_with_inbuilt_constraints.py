# In this example, we show how to use built-in prior constraints.

import bilby.core.prior
from bilby.core.prior import Constraint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import redback

# Import the constraints function only if you want custom bounds.
from redback.constraints import csm_constraints


model = 'csm_interaction'
function = redback.model_library.all_models_dict[model]

# The recommended path is to ask redback to attach the built-in constraints.
priors = redback.priors.get_priors(model=model, constraint=True)

# If you want custom constraint bounds instead, construct the PriorDict manually.
custom_priors = bilby.core.prior.PriorDict(conversion_function=csm_constraints)
custom_priors.update(redback.priors.get_priors(model=model))
custom_priors['shock_time'] = Constraint(0.6, 0.8)
custom_priors['photosphere_constraint_1'] = Constraint(0, 1)
custom_priors['photosphere_constraint_2'] = Constraint(0, 0.5)

# Please keep in mind that if you sample with fixed parameters that are required in the constraints function,
# you will get an error. You will need to sample with these parameters as well, or modify the constraints function yourself.

priors['kappa'] = 0.34
priors['redshift'] = 0.16
samples = pd.DataFrame(priors.sample(20))
time = np.linspace(200, 900, 500)

redshift = 0.01
for x in range(len(samples)):
    kwargs = samples.iloc[x]
    kwargs['output_format'] = 'magnitude'
    kwargs['bands'] = ['lsstg']
    mag = function(time, **kwargs)
    plt.plot(time, mag)
plt.ylim(21, 30)
plt.gca().invert_yaxis()
plt.show()
