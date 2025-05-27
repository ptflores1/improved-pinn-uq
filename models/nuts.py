from pyro.infer import NUTS as PyroNUTS

# We wrap the Pyro NUTS sampler to avoid reset the sampler after each run
# in this way we can use the same sampler for multiple runs 
# we do this to avoid keeping in memory all the samples
class NUTS(PyroNUTS):
    def cleanup(self):
        pass