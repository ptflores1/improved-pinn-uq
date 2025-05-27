from neurodiffeq.conditions import BaseCondition
import torch
import numpy as np


class CustomCondition(BaseCondition):
    r"""A custom condition where the parametrization is custom made by the user to comply with
    the conditions of the differential system.
    :param parametrization:
        The custom parametrization that comlpies with the conditions of the differential system. The first input
        is the output of the neural network. The rest of the inputs of the parametrization are
        the inputs to the neural network with the same order as in the solver.
    :type parametrization: callable
    """

    def __init__(self, parametrization):
        super().__init__()
        self.parameterize = parametrization

    def enforce(self, net, *coords):
        r"""Enforces this condition on a network with `N` inputs
        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param coordinates: Inputs of the neural network.
        :type coordinates: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`
        .. note::
            This method overrides the default method of ``neurodiffeq.conditions.BaseCondition`` .
            In general, you should avoid overriding ``enforce`` when implementing custom boundary conditions.
        """

        def ANN(*inner_coords):
            r"""The neural netowrk as a function
            :param coordinates: Inputs of the neural network.
            :type coordinates: `torch.Tensor`.
            :return: The output or outputs of the network.
            :rtype: list[`torch.Tensor`, `torch.Tensor`,...] or `torch.Tensor`.
            """
            if len(inner_coords) > 1:
                outs = net(torch.cat([*inner_coords], dim=1))
            else:
                outs = net(inner_coords[0])
            out_units = net.output_features
            if out_units > 1:
                outs = [outs[:, index].view(-1, 1) for index in range(out_units)]
            return outs

        return self.parameterize(ANN, *coords)

class reparam_CPL:
    def __init__(self, w_0=None, w_1=None) -> None:
        self.w_0 = w_0
        self.w_1 = w_1

    def __call__(self, ANN, z, w_0=None, w_1=None):
        w_0 = self.w_0 if w_0 is None else w_0
        w_1 = self.w_1 if w_1 is None else w_1
        x_1, x_2 = ANN(z)
        z_0s = torch.zeros_like(z, requires_grad=True)
        x_1_0, x_2_0 = ANN(z_0s)

        out = np.e**(3*(1 + w_0)*(x_1 - x_1_0) + 3*w_1*(x_2 - x_2_0))
        return out

class quint_reparams:
    def __init__(self, N_0_abs, lam_prime, Om_m_0):
        self.N_0_abs = torch.tensor(N_0_abs)
        self.lam_prime = torch.tensor(lam_prime) if lam_prime is not None else None
        self.Om_m_0 = torch.tensor(Om_m_0) if Om_m_0 is not None else None

    def x_reparam(self, ANN, N_prime, lam_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the first dependent variable
        of the differential system of the quintessence model,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{x}\left(N^{\prime}, \lambda^{\prime}, \Omega_{m,0}^{\Lambda}\right)=
        \left(1-e^{-N^{\prime}}\right)\left(1-e^{-\lambda^{\prime}}\right)
        x_{\mathcal{N}}\left(N^{\prime}, \lambda^{\prime}, \Omega_{m,0}^{\Lambda}\right).`
        :param ANN: The neural network.
        :type ANN: function.
        :param N_prime: The independent variable.
        :type N_prime: `torch.Tensor`
        :param lam_prime: The first parameter of the bundle.
        :type lam_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        if lam_prime is None:
            x_N = ANN(N_prime) if callable(ANN) else ANN
        else:
            x_N = ANN(N_prime, lam_prime, Om_m_0) if callable(ANN) else ANN
        
        lam_prime = self.lam_prime.to(x_N.device) if lam_prime is None else lam_prime
        Om_m_0 = self.Om_m_0.to(x_N.device) if Om_m_0 is None else Om_m_0

        out = (1 - torch.exp(-N_prime)) * (1 - torch.exp(-lam_prime)) * x_N
        return out

    def y_reparam(self, ANN, N_prime, lam_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the second dependent variable
        of the differential system of the quintessence model,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{y}\left(N^{\prime}, \lambda^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{y}\left(N^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-N^{\prime}}\right)\left(1-e^{-\lambda^{\prime}}\right)
        y_{\mathcal{N}}\left(N^{\prime}, \lambda^{\prime}, \Omega_{m,0}^{\Lambda}\right),`
        where :math:`\hat{y}` is:
        :math:`\displaystyle
        \hat{y}\left(N^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        \sqrt{\dfrac{\left(1-\Omega_{m,0}^{\Lambda}\right)}{\Omega_{m,0}^{\Lambda}e^{-3\left|N_0\right|\left(N^{\prime}-1\right)}+1-\Omega_{m,0}^{\Lambda}}}.`
        :param ANN: The neural network.
        :type ANN: function.
        :param N_prime: The independent variable.
        :type N_prime: `torch.Tensor`
        :param lam_prime: The first parameter of the bundle.
        :type lam_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        if lam_prime is None:
            y_N = ANN(N_prime) if callable(ANN) else ANN
        else:
            y_N = ANN(N_prime, lam_prime, Om_m_0) if callable(ANN) else ANN

        lam_prime = self.lam_prime.to(y_N.device) if lam_prime is None else lam_prime
        Om_m_0 = self.Om_m_0.to(y_N.device) if Om_m_0 is None else Om_m_0

        N = (N_prime - 1)*self.N_0_abs.to(y_N.device)
        y_hat = ((1 - Om_m_0)/(Om_m_0*(np.e**(-3*N)) + 1 - Om_m_0)) ** (1/2)

        out = y_hat + (1 - torch.exp(-N_prime)) * (1 - torch.exp(-lam_prime)) * y_N
        return out


class HS_reparams:
    def __init__(self, z_0, b_prime_min, alpha, b_prime, Om_m_0):
        self.z_0 = z_0
        self.b_prime_min = b_prime_min
        self.alpha = alpha
        self.b_prime = torch.tensor(b_prime)
        self.Om_m_0 = torch.tensor(Om_m_0)

    def x_reparam(self, ANN, z_prime, b_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the first dependent variable,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{x}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        x_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right).`
        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha

        if b_prime is None:
            x_N = ANN(z_prime) if callable(ANN) else ANN
        else:
            x_N = ANN(z_prime, b_prime, Om_m_0) if callable(ANN) else ANN

        b_prime = self.b_prime.to(x_N.device) if b_prime is None else b_prime
        Om_m_0 = self.Om_m_0.to(x_N.device) if Om_m_0 is None else Om_m_0

        out = (1 - torch.exp(-z_prime)) * (1 - torch.exp(-alpha*(b_prime - self.b_prime_min))) * x_N
        return out

    def y_reparam(self, ANN, z_prime, b_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the second dependent variable,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{y}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{y}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        y_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`
        where :math:`\hat{y}` is:
        :math:`\displaystyle
        \hat{y}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        \dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}
        +2\left(1-\Omega_{m,0 }^{\Lambda}\right)}{2\left[\Omega_{m,0}^{\Lambda}
        \left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}+1-\Omega_{m,0}^{\Lambda}\right]}.`
        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        if b_prime is None:
            y_N = ANN(z_prime) if callable(ANN) else ANN
        else:
            y_N = ANN(z_prime, b_prime, Om_m_0) if callable(ANN) else ANN

        b_prime = self.b_prime.to(y_N.device) if b_prime is None else b_prime
        Om_m_0 = self.Om_m_0.to(y_N.device) if Om_m_0 is None else Om_m_0

        y_hat = (Om_m_0*((1 + z)**3) + 2*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = y_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(-alpha*(b_prime - self.b_prime_min))) * y_N
        return out

    def v_reparam(self, ANN, z_prime, b_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the third dependent variable,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{v}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{v}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        v_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`
        where :math:`\hat{v}` is:
        :math:`\displaystyle
        \hat{v}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        =\dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}
        +4\left(1-\Omega_{m,0 }^{\Lambda}\right)}{2\left[\Omega_{m,0}^{\Lambda}
        \left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}+1-\Omega_{m,0}^{\Lambda}\right]}.`
        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        if b_prime is None:
            v_N = ANN(z_prime) if callable(ANN) else ANN
        else:
            v_N = ANN(z_prime, b_prime, Om_m_0) if callable(ANN) else ANN

        b_prime = self.b_prime.to(v_N.device) if b_prime is None else b_prime
        Om_m_0 = self.Om_m_0.to(v_N.device) if Om_m_0 is None else Om_m_0

        v_hat = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = v_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(-alpha*(b_prime - self.b_prime_min))) * v_N
        return out

    def Om_reparam(self, ANN, z_prime, b_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the fourth dependent variable,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{\Omega}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{\Omega}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        \Omega_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`
        where :math:`\tilde{\Omega}` is:
        :math:`\displaystyle
        \tilde{\Omega}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}}
        {\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}+1-\Omega_{m,0}^{\Lambda}}.`
        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        if b_prime is None:
            Om_N = ANN(z_prime) if callable(ANN) else ANN
        else:
            Om_N = ANN(z_prime, b_prime, Om_m_0) if callable(ANN) else ANN

        b_prime = self.b_prime.to(Om_N.device) if b_prime is None else b_prime
        Om_m_0 = self.Om_m_0.to(Om_N.device) if Om_m_0 is None else Om_m_0

        Om_hat = Om_m_0*((1 + z)**3)/((Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = Om_hat + (1 - torch.exp(-z_prime)) * (1 - torch.exp(-alpha*(b_prime - self.b_prime_min))) * Om_N
        return out

    def r_prime_reparam(self, ANN, z_prime, b_prime=None, Om_m_0=None):
        r"""The reparametrization of the output of the netowork assinged to the fifth dependent variable,
        which in this case corresponds to a perturbative reparametrization:
        :math:`\displaystyle \tilde{r}^{\prime}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda} \right)=
        \hat{r}^{\prime}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)
        +\left(1-e^{-z^{\prime}}\right)\left(1-e^{-\alpha b^{\prime}}\right)
        r^{\prime}_{\mathcal{N}}\left(z^{\prime}, b^{\prime},\Omega_{m,0}^{\Lambda}\right),`
        where :math:`\hat{r}^{\prime}` is:
        :math:`\displaystyle
        \hat{r}^{\prime}\left(z^{\prime},\Omega_{m,0}^{\Lambda}\right)=
        \dfrac{\Omega_{m,0}^{\Lambda}\left(1+z_{0}\left(1 - z^{\prime}\right)\right)^{3}
        +4\left(1-\Omega_{m,0 }^{\Lambda}\right)}{1-\Omega_{m,0}^{\Lambda}}.`
        :param ANN: The neural network.
        :type ANN: function.
        :param z_prime: The independent variable.
        :type z_prime: `torch.Tensor`
        :param b_prime: The first parameter of the bundle.
        :type b_prime: `torch.Tensor`
        :param Om_m_0: The second parameter of the bundle.
        :type Om_m_0: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`.
        """
        alpha = self.alpha
        z = self.z_0 * (1 - z_prime)

        if b_prime is None:
            r_prime_N = ANN(z_prime) if callable(ANN) else ANN
        else:
            r_prime_N = ANN(z_prime, b_prime, Om_m_0) if callable(ANN) else ANN

        b_prime = self.b_prime.to(r_prime_N.device) if b_prime is None else b_prime
        Om_m_0 = self.Om_m_0.to(r_prime_N.device) if Om_m_0 is None else Om_m_0

        r_hat = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(1 - Om_m_0)

        if isinstance(r_hat, torch.Tensor):
            r_prime_hat = torch.log(r_hat)
        else:
            r_prime_hat = np.log(r_hat)

        out = r_prime_hat + (1 - torch.exp(-z_prime)
                             ) * (1 - torch.exp(-alpha*(b_prime - self.b_prime_min))) * r_prime_N
        return out
