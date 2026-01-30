import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from PhagoPred.survival_analysis.data import synthetic_data

def plot_test(length=1000, strength=1000, sigma=30):
    test_hazards = np.zeros(length)
    # test_hazards += synthetic_data.get_gaussian_curve(length, center=np.random.rand()*length, sigma=sigma) * np.random.rand()
    # test_hazards[200:] += 0.1
    # test_hazards += synthetic_data.get_gaussian_curve(length, center=np.random.rand()*length, sigma=sigma) * np.random.rand()
    test_hazards += synthetic_data.get_gaussian_curve(length, 500, sigma=sigma) * strength
    pmf = synthetic_data.pmf_from_hazards(test_hazards)
    # print(pmf.sum())
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.plot(test_hazards, label='Hazards')
    # plt.plot(pmf, label='PMF')
    # plt.title('Hazard Function')
    # plt.xlabel('Time')
    # plt.ylabel('Hazard')
    # plt.legend()
    # plt.savefig(Path('temp') / 'hazard_pmf_test.png')
    return pmf.sum()
    
def plot_pmf_sum_strength():
    strengths = np.arange(1e-3,10, 1e-2)
    # for sigma in tqdm(range(10, 100, 10)):
    pmf_sum = []
    for strength in strengths:
        pmf_sum.append(plot_test(1000, strength, 20))

    plt.plot(strengths, pmf_sum, label=30)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Gaussian strength multiplier')
    plt.ylabel('CIF at last time step')
    plt.savefig(Path('temp') / 'hazard_pmf_test.png')
    
    
if __name__ == "__main__":
    # plot_test()
    plot_pmf_sum_strength()
    