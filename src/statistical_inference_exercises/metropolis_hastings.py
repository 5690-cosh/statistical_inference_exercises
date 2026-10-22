#mh
import numpy as np
import scipy
from src.statistical_inference_exercises.models import environ
 


def MH(ground_truth:np.ndarray, measurements: np.ndarray, theta_0: np.ndarray, proposal_params:tuple, num_iterations:int,s,t):

    theta_samples = [np.zeros_like(theta_0)]*num_iterations
    theta_samples[0] = theta_0
    accepted_list = []

    def proposal(mu:float|np.ndarray,sigma:float, shape:tuple): 
        #assume normal distribution for proposal (instrumental) distribution
        # draw a candidate from this distribution
        #candidate = theta_0 + np.random.normal(loc = mu, scale = sigma , size =shape )
        candidate = np.random.normal(loc = mu, scale = sigma , size =shape )
       
        return candidate

    def target_distribution(theta):
        new_model = environ([theta[0],0.07,1.505,30.1525],s,t)
        predictions = new_model.run()
        loglikelihood = np.sum(scipy.stats.norm.logpdf(measurements, loc=predictions , scale = 1))
        logprior = scipy.stats.uniform.logpdf(theta, loc = 7, scale = 6)
        #logprior = scipy.stats.norm.logpdf(theta, loc = 10, scale = 1)
        return loglikelihood+logprior

    def acceptance_ratio(n,theta_prime,mu,sigma):
        candidate_q_prob = scipy.stats.norm.logpdf(theta_prime, loc = mu, scale = sigma)
        current_q_prob = scipy.stats.norm.logpdf(theta_samples[n-1], loc = theta_prime, scale = sigma)
        candidate_target_prob = target_distribution(theta_prime)
        current_target_prob = target_distribution(theta_samples[n-1])
        ratio = (candidate_target_prob - current_target_prob)+(current_q_prob-candidate_q_prob)
        #ratio = np.exp((candidate_target_prob - current_target_prob)+(current_q_prob-candidate_q_prob))
        return ratio


    for n in range(1,num_iterations):
        theta_prime = proposal(theta_samples[n-1],proposal_params[1], theta_0.shape)

        #calculated using logs
        alpha = acceptance_ratio(n,theta_prime, theta_samples[n-1], proposal_params[1])
        u = np.random.random_sample(1) # draw from uniform distibrution U[0,1)
        condition = np.log(u) <= np.minimum(0, alpha)
        #condition = u <= np.minimum(1, alpha)
        accepted_list.append(condition)
        if (condition):
            theta_samples[n] = theta_prime
        else:
            theta_samples[n] = theta_samples[n-1]

    return theta_samples, accepted_list


if __name__ == "__main__":
    print("This is the metropolis_hastings.py file.")