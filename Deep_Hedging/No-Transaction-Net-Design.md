# Env
- `Market`
## Observation Space
: `spaces.Dict`
=> TD3기반 모델을 쓴다면 `Actor.feature_extract()`에서 `preprocess_obs()`를 호출하고, `Dict` space는 각각 key에 할당된 entity들을 따로 preprocess하므로
`feature_extractor`에 여러 obs를 넣을 수 있다.

