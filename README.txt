This repo contains an actor critic reinforcement learning algorithm set up to train an agent as to when
to close a position. Sample data is provided for ETHUSD for 1 day, April 1st 2019 (Bitmex, sec, resolution). The
sample data contains out-of-sample probabilities whether the current second is a significant peak or low, previously trained with LightGBM.
Given the current parameters, this results in 50 entries. The benchmark is an algorithm that exits based on these
probabilities, MA, EMA and attained.
The goal is to achieve a higher profit with the agent. Currently, the training a successful policy fails. Suggestions are welcome and I'll be happy to share the entire algorithm which may become profitable given a useful alpha for entering. 