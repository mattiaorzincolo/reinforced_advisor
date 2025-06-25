# Robo-advisor for Portfolio Optimization
## How Reinforcement Learning and LSTM net can combine for great insights

This project investigates an application of Reinforcemente Learning in the Finance industry and it is the code backbone of the MSc degree thesis of Mattia Orzincolo, laureate at Ca' Foscari University of Venice (2025).

The whole code is a developement of the original project "portfolio-optimization" by Noufal Samsudin, Data Scientist.

---

### Theorical Background

Since Artificial Intelligence applications are spreading in the finance industry, this project addresses how to harness AI decision-making models to provide advice for investors.  

Quantitative models for portfolio management mostly rely on the Modern Portfolio Theory with proven profits but also constraints. Inspired by recent literature on Reinforcement Learning (RL) agents as Robo advisors, I developed a tailor-made RL model for dynamic portfolio optimization enhanced by a Long Short Term Memory (LSTM).  

Reinforcement Learning is a model-free algorithm, which means that during the training it adapts its policy to maximize the reward function (portfolio return) by capturing the recurring patterns of the price time series. Conversely, Markowitz only provide an trade-off between volatility and return according to past observation, without any comprehension of the underlying patterns which leads to inaccurate advice especially in time series with great ups & downs.

