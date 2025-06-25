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

Reinforcement Learning has a complete different approach. This model actively interact with the environment (the market), which send observation to an Agent one timestep at time. The Agent takes an action according with the obseravtion it gets, trying to maximize the value of a reward function:
<div align='center'>
<img width="274" alt="rl" src="https://github.com/user-attachments/assets/81fb4b33-22af-4a07-a1d0-b928b46e22b0" />
</div>

