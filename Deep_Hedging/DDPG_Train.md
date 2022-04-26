# DDPG Train
`DDPG.learn()` -> ... -> `OffPolicyAlgorithm.train()` -> `TD3.train()`  
`TD3.train()`~ `TD3Policy.actor.forward()`->`Actor.forward()`->`TD3Policy.features_extractor.forward()`

=> 즉, DDPG의 policy 단계에서 `feature_extractor`를 조정해서 `observation`에 대한 processing을 control 할 수 있다.