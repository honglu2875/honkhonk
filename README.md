# honkhonk

honk honk, honk honk honk honk!

# What's up
First, check out [demo/run_elm_2d.py](demo/run_elm_2d.py)

Ok, so everything above those `CONFIGSTORE.store(...)` was just to set up the configs. They are self-explanatory and all we may need to change are:
- in CustomModelConfig,
  - model_path
- in CustomMAPElitesConfig
  - map_grid_size (the number of bins. We are doing 1-dim behavior space determined by the prompt length)
  - init_steps (how many steps to do random generation)
  - total_steps (total_steps - init_steps is the number of steps to mutate)
in RedTeamingConfig
  - batch_size (the number of prompts generated per mutation)


In CustomPromptEvolution, a few methods are overwritten to implement what we are doing. A few things:
- The `self.fitness_model` is the model used to get the fitness score. So far it rates the sentiment by calling some model and then passing through `get_sentiment_score` function.
- A genotype is basically a bunch of prompt templates (static) plus
   - "instruction_str" (the current question)
- `mutate_prompt` function mutates a Genotype by asking the LM to rewrite a more harmful question. It uses `mutation_instructions` and inserts its `instruction_str` in it. A new instruction will be generated with which we create a new Genotype.
- `random_prompt` is the initial random generation. Now it calls `get_initial_prompts` to randomly create questions conditioned on the news article.
- `evaluate_string` is the function that generates LM response to the prompt. The response will be fed into `fitness` for toxicity evaluation.