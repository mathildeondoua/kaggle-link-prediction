# Agent Guidelines

## Code Quality & Execution
- Always run your code and check for errors before moving on
- Comment your code in english
- Write clean, modular code: separate data loading, feature engineering, modeling, and submission generation into clearly named functions or sections
- Use comments to explain non-obvious steps; name variables explicitly (no `x`, `tmp`, `df2`)


## Planning & Reasoning
- Think step by step; produce an explicit implementation plan before writing any code
- Be critical of your own plan: identify weak points, potential data leakage, and edge cases
- When exploring data, always form hypotheses first, then validate them with code

## Iteration & Improvement
- After each experiment, summarize what worked, what didn't, and why
- Propose concrete next steps based on evidence, not guesses
- Track results (metric scores) explicitly; never claim improvement without a number to back it up

## Constraints
- **IMPORTANT** Do not use deep learning models (no GNNs, no neural networks, no transformers)
- Focus on classical ML (gradient boosting, random forests, logistic regression, SVM, etc.)
- Prioritize feature engineering over model complexity