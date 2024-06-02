import torch
from scipy.stats import gaussian_kde
import numpy as np

def time_intervals(t, target_levels):

    # Time distribution
    # Estimate the PDF using KDE
    t_pdf = gaussian_kde(t)
    x = np.linspace(min(t), max(t), 100)
    t_pdf_values = t_pdf(x)
    t_pdf_values = t_pdf_values.reshape(x.shape)
    
    def find_credible_intervals(x, pdf_values, target_levels):
        dx = x[1] - x[0]
        cumulative = np.cumsum(pdf_values) * dx
        
        intervals = []
        for target in target_levels:
            left_idx = np.where(cumulative >= 0)[0][0]
            right_idx = np.where(cumulative >= min(max(cumulative),(target)))[0][0]
            
            intervals.append((0, x[right_idx]))
        
        return intervals
    
    # Find credible intervals for 68% and 95% levels
    intervals = find_credible_intervals(x, t_pdf_values, target_levels)
    return intervals, t_pdf, x, t_pdf_values

def loc_level(loc, target_levels):
    loc_pdf = gaussian_kde(loc.T)
    x = np.linspace(min(loc[:,0]), max(loc[:,0]), 100)
    y = np.linspace(min(loc[:,1]), max(loc[:,1]), 100)
    x, y = np.meshgrid(x, y)
    
    # Evaluate the PDF at the grid points
    loc_pdf_values = loc_pdf(np.vstack([x.ravel(), y.ravel()]))
    loc_pdf_values = loc_pdf_values.reshape(x.shape)
    
    
    def find_contour_levels(grid, target_levels):
        sorted_grid = np.sort(grid.ravel())
        total = sorted_grid.sum()
        cumulative = sorted_grid.cumsum()
        
        levels = []
        for target in target_levels[::-1]:
            idx = np.where(cumulative >= (1 - target) * total)[0][0]
            levels.append(sorted_grid[idx])
        
        return levels

    levels = find_contour_levels(loc_pdf_values, target_levels)
    return levels, loc_pdf, x, y, loc_pdf_values

class ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    Confidence outputs are divided into equally-sized interval bins. In each bin, we compute the confidence gap as:
    bin_gap = l1_norm(avg_confidence_in_bin - accuracy_in_bin)
    A weighted average of the gaps is then returned based on the number of samples in each bin.
    """

    def __init__(self, n_bins: int = 15):
        """
        :param n_bins: number of confidence interval bins.
        :param activation: callable function for logit normalisation.
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels, mode):  # type: ignore
        if mode == 'sample':
            predictions = torch.mode(probs, 1)[0]
            confidences = probs.eq(predictions.unsqueeze(-1)).sum(-1)/probs.size(1)
        else:
            confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        correct_list = []
        num_list = []
        ece = torch.zeros(1, device=labels.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):  # type: ignore
            # Calculated 'confidence - accuracy' in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            num_in_bin = in_bin.float().sum()
            if num_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * num_in_bin
                correct_list.append(accuracy_in_bin * num_in_bin)
                num_list.append(num_in_bin)
            else:
                correct_list.append(0.)
                num_list.append(0.)
        return ece, correct_list, num_list

def get_calibration_score(time_all, loc_all, mark_all, time_gt, loc_gt, target_levels = np.linspace(0.5, 0.9, 5), model='ddSMTPP'):
    ece, correct_list, num_list = 0, [], []
    if loc_gt.size(-1)==3:
        mark_gt = loc_gt[:,0]
        loc_gt = loc_gt[:,1:]
        if mark_all is not None:
            if model=='DSTPP':
                mark_scores = torch.max(torch.cat(mark_all,1), dim=-1)[1]
                mode = 'sample'
            else:
                mark_scores = torch.cat(mark_all,1).mean(1)
                mode = 'probs'
            # mark_scores /= mark_scores.sum(-1,keepdim=True)
            eceloss = ECELoss(n_bins=10)
            ece, correct_list, num_list = eceloss(mark_scores, (mark_gt-1).long(), mode=mode)
    

    time_samples = torch.cat(time_all,1)
    loc_samples = torch.cat(loc_all,1)
    # print(loc_samples.size())

    calibration_time = torch.zeros(len(target_levels))
    calibration_loc = torch.zeros(len(target_levels))
    for t, loc, t_g, loc_g in zip(time_samples, loc_samples, time_gt, loc_gt):
        intervals, t_pdf, _,_ = time_intervals(t, target_levels)
        calibration_time += (t_g >= torch.tensor([intervals[i][0] for i in range(len(intervals))])) & (t_g <= torch.tensor([intervals[i][1] for i in range(len(intervals))]))
        levels, loc_pdf, _,_, _ = loc_level(loc, target_levels)
        calibration_loc += loc_pdf(loc_g) >= np.array(levels[::-1])

    
    CS_time = torch.abs(calibration_time - target_levels * len(time_samples))
    CS_loc = torch.abs(calibration_loc - target_levels * len(time_samples))

    return [CS_time, CS_loc, calibration_time, calibration_loc, ece, torch.tensor(correct_list), torch.tensor(num_list)]