import torch
import torch.optim as optim
import numpy as np

class Cagrad_all:
    def __init__(self, c=0.5):
        self.cagrad_c = c

    def cagrad(self, grad_vec, num_tasks):
        """
        Solves the dual optimization problem to find the common descent direction.
        Args:
            grad_vec: Tensor of shape [num_tasks, total_params]
            num_tasks: Number of constraints/losses being merged
        """
        grads = grad_vec

        # Compute Gradient Gram Matrix (GG = G * G^T)
        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        # Optimization variable w (weight for each task)
        w = torch.zeros(num_tasks, 1, requires_grad=True)
        
        # Optimizer for the inner loop (Dual Ascent)
        # Using high LR as per original implementation logic
        
        w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c_const = (gg + 1e-4).sqrt() * self.cagrad_c

        w_best = None
        obj_best = np.inf
        
        # Optimization loop (20 iterations is standard, 1 extra for final evaluation)
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            
            # Objective function of the dual problem
            obj = ww.t().mm(Gg) + c_const * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            
            if i < 20:
                obj.backward()
                w_opt.step()

        # Compute final merged gradient using optimal weights
        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c_const.view(-1) / (gw_norm + 1e-4)
        
        # g = g_0 + lambda * g_w
        g_update = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + self.cagrad_c**2)
            
        return g_update