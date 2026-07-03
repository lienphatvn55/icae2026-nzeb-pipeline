import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, lambda_bound=0.1, lambda_mono=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_bound = lambda_bound
        self.lambda_mono = lambda_mono
        
    def forward(self, pred, target, features=None):
        """
        pred: Predicted EUI [batch, 1]
        target: Target physics EUI [batch, 1]
        features: Flat tensor of features used for monotonicity checks [batch, num_features]
                  Order must match the notebook FEATURE_NAMES (9 features, GROSS EUI target):
                  0:Wall_U, 1:Roof_U, 2:Roof_Ref, 3:Win_U, 4:Win_SHGC, 5:COP, 6:Cool_SP, 7:LPD, 8:DeltaT
        """
        # 1. Base MSE Loss
        loss_mse = self.mse(pred, target)
        
        # 2. Bounding Loss (EUI should be within realistic limits for NZEB office, e.g., 10 to 200)
        # Soft penalty for going out of bounds
        lower_bound = 10.0
        upper_bound = 200.0
        loss_bound = torch.mean(torch.relu(lower_bound - pred)) + torch.mean(torch.relu(pred - upper_bound))
        
        # 3. Monotonicity Loss (if features and gradients are available)
        loss_mono = torch.tensor(0.0, device=pred.device)
        
        if features is not None and features.requires_grad:
            # We want to penalize if the gradient w.r.t specific features violates physics
            # GROSS EUI should INCREASE with U-value (idx 0, 1, 3) -> grad should be > 0
            # GROSS EUI should INCREASE with SHGC (idx 4) -> grad should be > 0
            # GROSS EUI should INCREASE with DeltaT (idx 8) -> grad should be > 0
            # GROSS EUI should DECREASE with COP (idx 5) -> grad should be < 0

            # Compute gradients of predictions w.r.t features
            grads = torch.autograd.grad(
                outputs=pred,
                inputs=features,
                grad_outputs=torch.ones_like(pred),
                create_graph=True,
                retain_graph=True
            )[0]  # Shape: [batch, 9]

            # Penalize negative gradients for features that should increase EUI
            pos_features = [0, 1, 3, 4, 8]
            for idx in pos_features:
                loss_mono += torch.mean(torch.relu(-grads[:, idx]))

            # Penalize positive gradients for features that should decrease EUI
            neg_features = [5]
            for idx in neg_features:
                loss_mono += torch.mean(torch.relu(grads[:, idx]))
                
        total_loss = loss_mse + (self.lambda_bound * loss_bound) + (self.lambda_mono * loss_mono)
        
        return total_loss, loss_mse, loss_bound, loss_mono
