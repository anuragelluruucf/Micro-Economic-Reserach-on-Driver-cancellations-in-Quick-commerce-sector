# Micro-Economic-Reserach-on-Driver-cancellations-in-Quick-commerce-sector
This repository contains my academic research project on strategic cancellations in food delivery platforms. Delivery riders often accept an order and later cancel with unverifiable excuses such as “bike issues,” leading to delays, customer dissatisfaction, and hidden operational losses. The goal of this study was to investigate the causes of these cancellations, structure the problem using economic theory, and develop predictive models to mitigate them.

Objectives

Understand why and when riders cancel after accepting orders.

Apply economic theory (information asymmetry, moral hazard, discrete choice) to frame the problem.

Build predictive analytics models to detect and prevent strategic cancellations.

Propose fair and scalable interventions for real-world deployment.

Approach

Data Cleaning & Validation

Processed and cleaned 447,000+ delivery records using Linux command-line workflows.

Addressed missingness in timestamps and rider history fields.

Feature Engineering

Distance → delivery cost.

Session time → fatigue.

Peak hours → outside option value.

Repeated unverifiable excuses → proxy for hidden intent.

Modeling & Testing

Proxy labeling to identify strategic riders.

Random Forest classifiers with SHAP explainability.

Balanced sampling for class imbalance.

Hypothesis testing with logistic regression, t-tests, and Z-tests.

Policy Simulation

Designed risk-based interventions:

Low risk: normal processing.

Medium risk: photo verification.

High risk: callback/manual override.

Simulations showed up to 40% reduction in strategic cancellations.

Findings

Strategic cancellations make up ~1.3% of orders, caused by a small set of riders.

Key drivers: repetition of unverifiable excuses, longer distances, peak-hour timing.

Not predictive: cancellation speed after pickup.

Cold-start models achieved AUC ~0.68, showing risk can be detected even for new riders.

Estimated 1,100 staff-hours lost monthly due to strategic cancels, highlighting operational impact.

Conclusion

This research demonstrates that economic theory provides structure and analytics delivers solutions. By combining rigorous data cleaning, feature engineering, predictive modeling, and policy simulation, I was able to identify the true causes of strategic cancellations and propose actionable solutions. If implemented in practice, this framework could improve platform efficiency, protect fairness for genuine riders, and enhance customer trust.
