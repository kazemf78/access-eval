"""
Accessibility-profile metrics and their budget-limited estimation (the paper's definitions), with no plotting or
notebook dependencies so evaluation workflows can import them in a slim environment.

    metrics.per_inquiry(...)      graded rows -> per-inquiry y (original), c (correct paraphrases), m (paraphrases)
    metrics.profile_metrics(...)  acc, mu, R, L, B, reversal, Access@k / Reliable@k (+ bootstrap CIs)
    estimation.*                  budget designs vs. full-run references (paper Appendix J and the Phase I plan)
"""
