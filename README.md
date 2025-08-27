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

Reinforcement Learning has a complete different approach. This model actively interact with the environme
nt (the market), which send observation to an Agent one timestep at time. The Agent takes an action according with the obseravtion it gets, trying to maximize the value of a reward function:
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
he key contribution to this approach is the design of the Actor Network, which is responsible for defyning the actions to be taken, capture price movements and maximize the return of our portfolio.

The Actor Network defined inside the *class ActorRNNNetwork* has a Long Short Term Memory Layer of (64), a Dense layer (fully_connected) with dimensions defined in *actor_fc_layers*  and then normalize with a L2 regularized and a norm layer. In the early training, just a RNN and a Dense layer combined for the ActorRNNNetwork but the model suffered of vanishing gradients, typical issue of Recurrent Networks. LSTM are less prone to this problem and L2 togheter with Norm layer provied a more stabile training.


### Results
