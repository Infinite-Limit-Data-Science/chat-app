### Probabilities: why the results are not perfect

There are multiple ways to view probabilities. 

Let's start off with the basics. You may have some observed data. And you may want to learn whether whether the observed sample mean is so far from the hypothesized mean that its mean would be “unlikely” if the hypothesis were true. If it’s “too far,” you reject the hypothesis; otherwise, you fail to reject it. Now maybe the hypothesized mean is correct, and you just happened to get “bad” data (an extreme sample). In fact, that can happen, and classical hypothesis testing allows for that possibility. If your sample is unrepresentative (just by random chance or by poor sampling), you could see an extreme sample mean that leads you to reject the hypothesis. Classical hypothesis testing tries to control how often that happens (the Type I error rate) across many repeated experiments.

In classical stats testing, you may have a population mean, a population standard deviation and your own sample mean based on sample data. Using a standard normal distribution, you can define a confidence interval built around your sample mean bound by two z-scores defining the curve if it is a two-tail test. You can run a z-test to check if the hypothesized value lies within that interval. If it does, you generally fail to reject the hypothesis at the chosen significance level. This tells you whether your observed data (sample mean) is statistically consistent (not too far off) with the hypothesized mean, given a certain cutoff (e.g. 5% significance). We don’t strictly say “accurate” in the sense of “true or false”; rather, we say whether or not the difference is statistically significant. This is in essence a z-test or when you don't have the population standard deviation and you have a sample standard deviation, then a t-test, which uses the degree of freedom principle.

z-tests and t-tests don't decide the probability of the hypothesized mean being true or false. Instead, it is a statement about sample data under the assumption that the population mean is correct. Note the hypothesized mean is not the same as the true (unknown) population mean, since knowing the entire population mean is usually impossible.

In contrast to the frequentist tests above (z-tests, t-tests, p-values), in Bayesian inference, given the data, you want to know the probability the hypothesis is true. **Bayes’ theorem is the mathematical engine that flips from “the probability of data given a hypothesis” to “the probability of a hypothesis given the data.”** In the Bayes formula, you have the Posterior: the probability that the hypothesis is true after seeing the data. In everyday words: “Now that I have new information (data), how likely is it that my hypothesis is correct?” This is what you want to find out – your updated belief about the hypothesis given the evidence. The Posterior representation: P(Hypothesis ∣ Data).

You have the Likelihood value: P(Data ∣ Hypothesis). This is the probability of seeing this specific data if the hypothesis were true. In everyday words: “How well does the hypothesis ‘explain’ or ‘predict’ the data?” Technical detail: This is not the same as the probability of the hypothesis given the data; it’s the other way around.

You have the prior value: P(Hypothesis). Meaning: Your belief about the hypothesis before seeing the new data. In everyday words: “How likely did I think the hypothesis was before I collected any new evidence?” Why it matters: In Bayesian thinking, you always start with some prior belief or distribution (even if it’s very broad or “uninformative”), because you never come into a problem with zero assumptions. Example: If you’re testing for a rare disease, your prior might be “only 1% of people have it.” Or if you’re testing whether a coin is fair, maybe your prior is “most coins are fair,” so you start off with a strong prior that the coin has a 50% heads rate.

You have the Evidence (or Marginal Likelihood) value: P(Data). Meaning: The overall probability of observing the data under all possible hypotheses. In everyday words: “Out of all the ways I could explain the data, what is the total likelihood of seeing these data at all?” Why it’s in the denominator: It acts like a normalizing constant to ensure the result P( Hypothesis ∣ Data ) is a proper probability (i.e., sums or integrates to 1 when you consider all possible hypotheses).

$$
\underbrace{P(\text{Hypothesis} \mid \text{Data})}_{\text{Posterior}}
\;=\;
\frac{
  \underbrace{P(\text{Data} \mid \text{Hypothesis})}_{\text{Likelihood}}
  \;\times\;
  \underbrace{P(\text{Hypothesis})}_{\text{Prior}}
}{
  \underbrace{P(\text{Data})}_{\text{Evidence}}
}
$$

Short Example: Disease Testing
- Hypothesis = “Person has the disease.”
- Prior = Prevalence (say, 1%).
- Likelihood = Probability of a positive test result if the person does have the disease (test sensitivity), and if they don’t (false positive rate).
- Posterior = Probability the person has the disease given that they tested positive.