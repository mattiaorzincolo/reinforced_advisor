# Robo-advisor for Portfolio Optimization
## How Reinforcement Learning and LSTM net can combine for great insights

This project investigates an application of Reinforcemente Learning in the Finance industry and it is the code backbone of the MSc degree thesis of Mattia Orzincolo, laureate at Ca' Foscari University of Venice (2025).

The whole code is a developement of the original project "portfolio-optimization" by Noufal Samsudin, Data Scientist.

---

### Theorical Background

Since Artificial Intelligence applications are spreading in the finance industry, this project addresses how to harness AI decision-making models to provide advice for investors.  

Inspired by the contribution of Wang & Yu (2021) that presented a tech solution for Robo-Advising, I developed a tailor-made Reinforcement Learning model for dynamic portfolio optimization enhanced by a Long Short Term Memory (LSTM).  

Reinforcement Learning is a model-free algorithm, which means that during the training it adapts its policy to maximize the reward function (portfolio return) by capturing the recurring patterns of the price time series. Conversely, Markowitz only provide an trade-off between volatility and return according to past observation, without any comprehension of the underlying patterns which leads to inaccurate advice especially in time series with great ups & downs.

<div align="center">
<img width="274" alt="markowitz" src="https://github.com/user-attachments/assets/5303db1c-09cd-4842-8383-c79a57ecd492" />
</div>

Where:
* $\\sum$ is the covariance matrix of returns
* $w$ refers to portfolio weights
* $K$ is the risk tolerance of the investor
* and $\\sum_{n}(w_{n})=1$ is the budget constraint
  

Markowitz Portfolio Theory generates potentially inifinite combinations of portfolio that obey to the above equations. For the sake of the analysis, I only considerate the Maximum Sharpe Portfolio, in other words the portfolio that is associated with the maximum Sharpe Ratio among all.

The Sharpe Ratio is a widely used measure in finance to assess the risk-adjusted return of an investment. It helps investors understand the return they receive for each unit of risk they take. A higher Sharpe Ratio generally indicates better risk-adjusted performance.
The Sharpe Ratio ($S_R$) is calculated as:

$$S_R = \frac{R_p - R_f}{\sigma_p}$$

Where:
* $R_p$ is the expected return of the portfolio or asset.
* $R_f$ is the risk-free rate of return (e.g., the return on a U.S. Treasury bond).
* $\sigma_p$ is the the standard deviation of the portfolio's or asset's excess return (i.e., the standard deviation of $(R_p - R_f)$). This measures the volatility or total risk of the investment.

### Framework: Reinforcement Learning equipped with a Deep Deterministic Policy Gradient (DDPG) algorithm

Reinforcement Learning has a complete different approach. This model actively interact with the environment (the market), which sends observation to an Agent one timestep at time. The Agent takes an action according with the obseravtion it gets, trying to maximize the value of a reward function:
<div align='center'>
<img width="274" alt="rl" src="https://github.com/user-attachments/assets/81fb4b33-22af-4a07-a1d0-b928b46e22b0" />
</div>

The final goal of any Reinforcement Learning model is to map a policy $\pi$ that represents the best action to take at a given state, according to the experience gained across the training. For more info about policy exploration and exploitation, Markov Decision Process and all other core features of a RL model, see *Sutton, R.S. and Barto, A.G. (2020) Reinforcement learning: An introduction. Cambridge, MA: The MIT Press*.  

This RL model has an actor-critic framework, more technically is a deep deterministic policy gradient (DDPG) algorithm with an off-policy actor-critic method for environments with a continuous action-space. It features a target actor and critic as well as an experience buffer. DDPG agents supports offline training (training from saved data, without an environment).

Key elements of a DDPG framework:

* Actor 𝜋(𝑆, 𝜃): the actor, with parameters 𝜃 , takes observation 𝑆 and returns the corresponding action that maximizes the long-term reward;

* Target actor $\pi_t$(𝑆, $\theta_t$): to improve the stability of the optimization, the agent periodically updates the target actor learnable parameters $\phi_t$ using the latest actor parameter values;

* Critic 𝑄(𝑆, 𝐴,𝜙): the critic, with parameters 𝜙 , takes observation 𝑆 and action 𝐴 as inputs and returns the corresponding expectation of the long-term reward;

* Target critic $Q_t$(𝑆, 𝐴, $\phi_t$): to improve the stability of the optimization, the agent periodically updates the target critic learnable parameters $\phi_t$ using the latest critic parameter values. 

The Deterministic Policy Gradient becomes "Deep" when it relies on Deep Neural Network to approximate the parameters $\theta$ and $\phi$ of the Actor and the Critic. 
The key contribution to this approach is the design of the Actor Network, which is responsible for defyning the actions to be taken, capture price movements and maximize the return of our portfolio.

The Actor Network defined inside the *class ActorRNNNetwork* has a Long Short Term Memory Layer of (64), a Dense layer (fully_connected) with dimensions defined in *actor_fc_layers*  and then regularized with a L2 Reg and a norm layer. In the early training, just a RNN and a Dense layer combined for the ActorRNNNetwork but the model suffered of vanishing gradients, typical issue of Recurrent Networks. LSTM are less prone to this problem and L2 togheter with Norm layer provied a more stabile training.

### Hyperparameters explained

Summary list of hyperparameters:
* Episode length = 1500
* N. Iterations = 500
* Number of environment steps collected per training iteration for the replay buffer = 100
* Log interval = 10
* Eval interval, iterations at which the policy is evaluated = 4
* Model save frequency = 12
* Replay buffer maximum length = 100000
* Batch size = 100
* Eval episodes = 4
* Standard deviation of the Ornstein-Uhlenbeck (OU) noise = 0.2
* Damping factor for the OU noise = 0.15
* 𝜏 = 0.05  Soft update
* 𝛾 = 0.05 Discount factor for future rewards
* Actor learning rate = 0.001
* Critic learning rate = 0.001 

---

### Dataset

The dataset contains time series with relevant trading information for three coins: DASH, Litecoin (LTC) and Stellar Lumens (STR or XLM in the updated version). For an effective training this kind of model requires a large dataset. Indeed, it is quite large storing exactly two years of 5 minutes stock price for combined 617k observations ranging from 2015-06-30 13:00 to 2017-06-30 13:00. 
Here the summary stats of the asset in the training phase, crypto market can express massive volatility:

<div align='center'>
<img width="500" height="500" alt="asset_stats" src="https://github.com/user-attachments/assets/bd0f1d88-f30c-4738-aa0b-16bc6b00fa5f" />
</div>

### Results

The first evidence in favour of this tailor made RL model is the learning curve of the training error. Assuming a 'vanilla' agent as an equivalent model to the proposed one, with the only difference in the design of the Actor Network: 'vanilla' has a simplier network, made just of a fully connected layer without any other regularization methods nor LSTM layers.
While the 'vanilla' learning seems steady across the training, the custom model significally grows after the 400 timestamp. Remember, this is not actually a training error since there is a reward function instead of a loss, so the more the better.

<div align='center'>
<img width="500" height="500" alt="training_line_chart" src="https://github.com/user-attachments/assets/eb1b0b86-a65c-4b06-a213-eed9f93b844b" />
</div>

In order to prove the overall dominance of a Reinforcement Learning model in asset allocation, three policies are tested at the same time: **RL+LSTM, Vanilla RL and Maximum Sharpe Portfolio**.

The RL+LSTM overperformed both the market as other two policies with a cumulative return in the period of **+58,00%** against negative performances of **–44,34%** of the Vanilla RL **–46,21%** of the Maximum Sharpe Portfolio policy.

<div align='center'>
<img width="500" height="500" alt="test" src="https://github.com/user-attachments/assets/93aa0ab7-c27f-451a-bf14-bbdd3a8a6e78" />
</div>

### Different Asset Allocation choices

#### Custom (RL+LSTM) policy
<div align='center'>
<img width="500" height="500" alt="actions_RL_custom" src="https://github.com/user-attachments/assets/aaf110f3-0b65-4575-b48c-f423099f2550" /><br>
</div>

#### Vanilla policy
<div align='center'>
<img width="500" height="500" alt="actions_vanilla" src="https://github.com/user-attachments/assets/15f09c6f-0627-40ae-8f11-5222ccbecc49" />
</div>

#### Markowitz policy
<div align='center'>
<img width="500" height="500" alt="actions_markowitz" src="https://github.com/user-attachments/assets/8f1ea85c-e6d7-4137-8d8a-f2c88f9e4db8" />
</div>

### References

<h5> Managing Reinforcement Learning within a continuous environment </h5>

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N. M., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015).  
*Continuous control with deep reinforcement learning*.  
arXiv:1509.02971. [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)

<h5> Evidence of using Recurrent Neural Networks for Portfolio selection </h5>

Lin, C.-M., Yeh, S.-N., Lee, C.-H., & Lee, C.-C. (2006).  
*Recurrent neural network for dynamic portfolio selection*.  
Applied Mathematics and Computation, 175(2), 1139–1146.  
doi: [10.1016/j.amc.2005.08.031](https://doi.org/10.1016/j.amc.2005.08.031)

<h5> Markowitz model explained </h5>

Markowitz, H. (1952).  
*Portfolio selection*.  
The Journal of Finance, 7(1), 77–91.  
doi: [10.2307/2975974](https://doi.org/10.2307/2975974)

<h5> Well-known toy model for Reinforcement Learning </h5>

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013).  
*Playing Atari with deep reinforcement learning*.  
arXiv preprint arXiv:1312.5602.  
[https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)

<h5> Exhaustive explaination about Reinforcement Learning </h5>

Sutton, R. S., & Barto, A. G. (2020).  
*Reinforcement learning: An introduction* (2nd ed.).  
Cambridge, MA: MIT Press.  
[http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

<h5> Greatest inspirations for this work </h5>

Wang, H., & Yu, S. (2021).  
*Robo-Advising: Enhancing Investment with Inverse Optimization and Deep Reinforcement Learning*.  
In *2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)* (pp. 365–372). IEEE.  
doi: [10.1109/ICMLA52953.2021.00063](https://doi.org/10.1109/ICMLA52953.2021.00063)

<h4> And a particular mention to Noufal Samsuding for providing explaination, code and data resulting as an effective baseline for my project. Take a look at his article on Medium </h4>

Samsudin, N. (2021).  
*Portfolio Optimization using Reinforcement Learning, Medium*. <br>
(https://medium.com/analytics-vidhya/portfolio-optimization-using-reinforcementlearning-1b5eba5db072)
