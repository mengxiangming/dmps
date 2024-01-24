from abc import ABC, abstractmethod
import torch
import numpy as np

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t

@register_conditioning_method(name='dmps') # DMPS method
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        alpha_bar = np.clip(alpha_bar, 1e-16, 1-1e-16)
        alpha_t = np.clip(alpha_t, 1e-16, 1-1e-16)
        scale_S = (1-alpha_bar)/(alpha_bar)

        S_vector = (1/(S*scale_S +noise_std**2)).to(x_t.device).reshape(-1,1)
        Temp_value = H_funcs.Ut(measurement - H_funcs.H(x_t)/np.sqrt(alpha_bar)).t()
        grad_value = H_funcs.Ht(H_funcs.U((S_vector*Temp_value).t()))
 
        grad_value = grad_value.reshape(x_t.shape)/np.sqrt(alpha_bar)
        x_t += self.scale*grad_value *(1-alpha_t)/np.sqrt(alpha_t)
        return x_t

