"""This module implements various loss functions for training machine learning
models.

It includes custom loss functions that extend PyTorch's nn.Module,
allowing for flexible and efficient computation of loss values during
training. The loss functions handle different scenarios such as
classification, regression, and segmentation tasks. They incorporate
techniques like weighted losses, focal losses, and smooth L1 losses to
address class imbalances and improve model performance. The module
ensures that the loss calculations are compatible with PyTorch's
autograd system, enabling seamless integration into training loops.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .normalizer import Normalizer


class ComposableLoss(nn.Module):
    """Compose the loss function using several terms. The importance of each
    term has to be specified in the configuration file. Each term with a >0
    weight will be added to the loss function.

    The loss term available are:
    - MSE: mean squared error between predicted and target normalized screen tags
    - MAE: mean absolute error between predicted and target normalized screen tags
    - JMSE: mean squared error between predicted and target J
    - JMAE: mean absolute error between predicted and target J
    - Pearson: Pearson correlation coefficient between predicted and target J
    - Fried: Fried parameter r0
    - Isoplanatic: Isoplanatic angle theta0
    - Rytov: Rytov variance sigma_r^2 that will be computed on log averaged Cn2
    - Scintillation_w: scintillation index for weak turbulence
    - Scintillation_m: scintillation index for moderate-strong turbulence

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration
    nz : Normalizer
        Normalizer object to be used to extract J in its original scale
    device : torch.device
        The device to use for the computation
    validation : bool
        If true, use the validation parameters from config
    """

    def __init__(self,
                 config: dict,
                 nz: Normalizer,
                 device: torch.device,
                 validation: bool = False):
        super(ComposableLoss, self).__init__()
        if validation:
            if 'val_loss' in config:
                config['loss'] = config['val_loss']
            else:
                print('[!] Warning: Validation loss not found in config.yaml,',
                      'keeping track of training loss instead')
        self.device = device
        self.loss_functions: dict[str, Callable] = {
            'MSE': torch.nn.MSELoss(),
            'MAE': torch.nn.L1Loss(),
            'JMSE': self._MSELoss,
            'JMAE': self._L1Loss,
            'Cn2MSE': self._do_nothing,
            'Cn2MAE': self._do_nothing,
            'Pearson': self._PearsonCorrelationLoss,
            'Fried': self._FriedLoss,
            'Isoplanatic': self._IsoplanaticLoss,
            'Rytov': self._RytovLoss,
            'Scintillation_w': self._ScintillationWeakLoss,
            'Scintillation_ms': self._ScintillationModerateStrongLoss,
        }
        self.loss_weights = {
            loss_name: config['loss'].get(loss_name, 0)
            for loss_name in self.loss_functions.keys()
        }
        self.total_weight = sum(self.loss_weights.values())
        self._select_loss_needed()

        # And get some useful parameters for the loss functions
        # the parameters are explained at:
        # https://males-project.github.io/SpeckleCn2Profiler/example/#configuration-file-explanation
        self.h = torch.Tensor([float(x) for x in config['speckle']['hArray']])
        self.k = 2 * torch.pi / (config['speckle'].get('lambda', 550) * 1e-9)
        self.cosz = np.cos(np.deg2rad(config['speckle'].get('z', 0)))
        self.secz = 1 / self.cosz
        self.p_fr = 0.423 * self.k**2 * self.secz
        self.p_iso = self.cosz**(8. / 5.) / ((2.91 * self.k**2)**(3. / 5.))
        self.p_scw = 2.25 * self.k**(7. / 6.) * self.secz**(11. / 6.)

        # We need to ba able to recover the tags
        self.recover_tag = nz.recover_tag
        # Move tensors to the device
        self.h = self.h.to(self.device)

    def _select_loss_needed(self):
        """Select the loss functions that will be used in the loss calculation.

        This allows to skip the computation of optional elements like
        Cn2.
        """
        self.loss_needed = {
            loss_name: loss_fn
            for loss_name, loss_fn in self.loss_functions.items()
            if self.loss_weights[loss_name] > 0
        }
        self.Cn2required = any([
            loss_name in [
                'Cn2MSE', 'Cn2MAE', 'Fried', 'Isoplanatic', 'Rytov',
                'Scintillation_w', 'Scintillations_ms'
            ] for loss_name in self.loss_needed.keys()
        ])

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass of the loss function.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags

        Returns
        -------
        loss : torch.Tensor
            The composed loss
        losses : dict
            Dictionary containing the individual losses
        """
        total_loss = 0
        losses = {}

        if self.Cn2required:
            Cn2_pred = self.reconstruct_cn2(pred)
            Cn2_target = self.reconstruct_cn2(target)

        for loss_name, loss_fn in self.loss_needed.items():
            weight = self.loss_weights[loss_name]
            if loss_name in ['MAE', 'MSE']:
                this_loss = loss_fn(pred, target)
            else:
                this_loss = loss_fn(pred, target, Cn2_pred, Cn2_target)
            total_loss += weight * this_loss
            losses[loss_name] = this_loss

        return total_loss / self.total_weight, losses

    def _do_nothing(*args, **kwargs):
        pass

    def get_J(self, Jnorm: torch.Tensor) -> torch.Tensor:
        """Recover J from the normalized tags. This needs to be done to compute
        Cn2.

        Parameters
        ----------
        Jnorm : torch.Tensor
            The normalized screen tags between 0 and 1

        Returns
        -------
        J : torch.Tensor
            The recovered screen tags
        """

        if Jnorm.ndim == 1:
            Jnorm = Jnorm[None, :]

        J = []
        for i in range(Jnorm.shape[0]):
            J.append(
                torch.tensor([
                    10**self.recover_tag[j](Jnorm[i][j], i)
                    for j in range(len(Jnorm[i]))
                ],
                             requires_grad=True).to(Jnorm.device))
        J = torch.stack(J)
        return J

    def reconstruct_cn2(self, Jnorm: torch.Tensor) -> torch.Tensor:
        """ Reconstruct Cn2 from screen tags
        c_i = J_i / (h[i+1] - h[i])

        Parameters
        ----------
        Jnorm : torch.Tensor
            The screen tags normalized between 0 and 1

        Returns
        -------
        Cn2 : torch.Tensor
            The Cn2 reconstructed from the screen tags, assuming a uniform profile
        """
        J = self.get_J(Jnorm)
        Cn2 = J / (self.h[1:] - self.h[:-1])
        return Cn2

    def _MSELoss(self,
                 pred: torch.Tensor,
                 target: torch.Tensor,
                 Cn2p: torch.Tensor = None,
                 Cn2t: torch.Tensor = None) -> torch.Tensor:
        """Mean squared error loss function.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The mean squared error loss
        """
        loss = self.loss_weights['JMSE'] * torch.mean(
            (pred - target)**2 / ((target**2).sum(-1).unsqueeze(-1) + 1e-5))
        # Optionally add the Cn2MSE loss
        if self.loss_weights['Cn2MSE'] > 0:
            loss += self.loss_weights['Cn2MSE'] * torch.mean(
                (Cn2p - Cn2t)**2 / ((Cn2t**2).sum(-1).unsqueeze(-1) + 1e-5))
        return loss

    def _L1Loss(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                Cn2p: torch.Tensor = None,
                Cn2t: torch.Tensor = None) -> torch.Tensor:
        """Mean absolute error loss function.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The mean absolute error loss
        """
        loss = self.loss_weights['JMAE'] * torch.mean(
            torch.abs(pred - target) / (target.sum(-1).unsqueeze(-1) + 1e-5))
        # Optionally add the Cn2MAE loss
        if self.loss_weights['Cn2MAE'] > 0:
            loss += self.loss_weights['Cn2MAE'] * torch.mean(
                torch.abs(Cn2p - Cn2t) / (Cn2t.sum(-1).unsqueeze(-1) + 1e-5))
        return loss

    def _PearsonCorrelationLoss(self, pred: torch.Tensor, target: torch.Tensor,
                                Cn2p: torch.tensor,
                                Cn2t: torch.Tensor) -> torch.Tensor:
        """Pearson correlation coefficient loss function.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The Pearson correlation coefficient loss
        """

        mean_pred = torch.mean(pred)
        mean_target = torch.mean(target)
        # Calculate covariance
        cov_xy = torch.mean((pred - mean_pred) * (target - mean_target))
        # Calculate standard deviations
        std_x = torch.std(pred)
        std_y = torch.std(target)
        # Calculate Pearson correlation coefficient
        corr = cov_xy / (std_x * std_y + 1e-5)
        # The loss is 1 - correlation to be minimized
        loss = 1.0 - corr

        return loss

    def get_FriedParameter(self, Jnorm: torch.Tensor) -> torch.Tensor:
        """Compute the Fried parameter r0 from the screen tags."""
        J = torch.Tensor(self.get_J(Jnorm))
        return (self.p_fr * torch.sum(J))**(-3 / 5)

    def _FriedLoss(self, pred: torch.Tensor, target: torch.Tensor,
                   Cn2p: torch.Tensor, Cn2t: torch.Tensor) -> torch.Tensor:
        """Fried parameter r0 loss function. The difference between real value
        and prediction is normalized by the real value, since small values are
        more important to get correctly.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The Fried parameter r0 loss
        """
        r0p = self.get_FriedParameter(pred)
        r0t = self.get_FriedParameter(target)
        loss = torch.abs(r0p - r0t) / r0t

        return loss

    def get_IsoplanaticAngle(self, Cn2: torch.Tensor) -> torch.Tensor:
        """Compute the isoplanatic angle theta0 from the screen tags."""
        # Integrate Cn2*z^(5/3)
        integral = torch.sum(
            Cn2 * (self.h[1:]**(8 / 3) - self.h[:-1]**(8 / 3))) * 3 / 8
        # Then I can compute theta0
        return self.p_iso / (integral**(3 / 5))

    def _IsoplanaticLoss(self, pred: torch.Tensor, target: torch.Tensor,
                         Cn2p: torch.Tensor,
                         Cn2t: torch.Tensor) -> torch.Tensor:
        """Isoplanatic angle theta0 loss function. The loss is normalized by
        the real value, since small values are more important to get correctly.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The isoplanatic angle theta0 loss
        """
        isp = self.get_IsoplanaticAngle(Cn2p)
        ist = self.get_IsoplanaticAngle(Cn2t)
        loss = torch.abs(isp - ist) / ist

        return loss

    def _RytovLoss(self, pred: torch.Tensor, target: torch.Tensor,
                   Cn2p: torch.Tensor, Cn2t: torch.Tensor) -> torch.Tensor:
        """Rytov variance sigma_r^2 loss function."""
        # //TODO: discuss about this. Since it is a variance measure,
        # it would have to be compared not to a single target,
        # but to the average of the dataset

        # throw a not implemented yet warning
        print('RytovLoss not implemented yet. Use scintillation index instead')
        return 0

    def get_ScintillationWeak(self, Cn: torch.Tensor) -> torch.Tensor:
        """Compute the scintillation index for weak turbulence sigma^2 from the
        screen tags."""
        # Integrate Cn2*z^(5/6)
        integral = torch.sum(
            Cn * (self.h[1:]**(11 / 6) - self.h[:-1]**(11 / 6))) * 6 / 11
        # Then I can compute sigma^2
        return self.p_scw * integral

    def _ScintillationWeakLoss(self, pred: torch.Tensor, target: torch.Tensor,
                               Cn2p: torch.Tensor,
                               Cn2t: torch.Tensor) -> torch.Tensor:
        """Scintillation index for weak turbulence loss function. The loss is
        normalized by the real value, since small values are more important to
        get correctly.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The scintillation index for weak turbulence loss
        """
        swp = self.get_ScintillationWeak(Cn2p)
        swt = self.get_ScintillationWeak(Cn2t)
        loss = torch.abs(swp - swt) / swt

        return loss

    def get_ScintillationModerateStrong(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the scintillation index for moderate-strong turbulence
        sigma^2 from the screen tags."""
        wsigma2 = self.get_ScintillationWeak(x)
        return torch.exp(wsigma2 * 0.49 / (1 + 1.11 * wsigma2**(6 / 5)) +
                         0.51 * wsigma2 / (1 + 0.69 * wsigma2**(6 / 5)))

    def _ScintillationModerateStrongLoss(self, pred: torch.Tensor,
                                         target: torch.Tensor,
                                         Cn2p: torch.Tensor,
                                         Cn2t: torch.Tensor) -> torch.Tensor:
        """Scintillation index for moderate-strong turbulence loss function.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        loss : torch.Tensor
            The scintillation index for moderate-strong turbulence loss
        """
        smp = self.get_ScintillationModerateStrong(Cn2p)
        smt = self.get_ScintillationModerateStrong(Cn2t)
        loss = torch.mean(torch.abs(smp - smt) / smt)

        return loss

    def _get_all_measures(self, pred: torch.Tensor, target: torch.Tensor,
                          Cn2p: torch.Tensor, Cn2t: torch.Tensor) -> dict:
        """Get all the measures available.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted screen tags
        target : torch.Tensor
            The target screen tags
        Cn2p : torch.Tensor
            The predicted Cn2
        Cn2t : torch.Tensor
            The target Cn2

        Returns
        -------
        measures : dict
            Dictionary containing the measures
        """
        measures = {}
        measures['Fried_true'] = self.get_FriedParameter(target)
        measures['Fried_pred'] = self.get_FriedParameter(pred)
        measures['Isoplanatic_true'] = self.get_IsoplanaticAngle(Cn2t)
        measures['Isoplanatic_pred'] = self.get_IsoplanaticAngle(Cn2p)
        measures['Scintillation_w_true'] = self.get_ScintillationWeak(Cn2t)
        measures['Scintillation_w_pred'] = self.get_ScintillationWeak(Cn2p)

        return measures
