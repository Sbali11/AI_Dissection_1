import torch


def expected_team_utility(x_probs, one_hot_y, alpha, beta, confidence, gamma, human_rationality=1):
    rational_rng = rng.uniform(0, 1)
    loss = (one_hot_y * torch.relu(x_probs - confidence) / (x_probs - confidence)) + (
        torch.relu(-x_probs + confidence) / (-x_probs + confidence) * (1 + beta) * alpha
        - beta
        - gamma
    )
    return loss


# function to calculate the team empirical utility
def empirical_team_utility(x_probs, y, human_rationality=1):
    rational_rng = rng.uniform(0, 1)
    if np.max(x_probs) >= confidence and rational_rng <= human_rationality:
        if np.argmax(x_probs) == y:
            return 1
        else:
            return -beta
    else:
        human = rng.uniform(0, 1)
        if human < alpha:
            return 1 - gamma
        else:
            return -beta - gamma
