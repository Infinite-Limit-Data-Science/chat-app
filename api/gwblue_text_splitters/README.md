# Inferencial Statistics

### Classical Statistics
There are multiple ways to view probabilities. 

Let's start off with the basics. You may have some observed data. And you may want to learn whether whether the observed sample mean is so far from the hypothesized mean that its mean would be “unlikely” if the hypothesis were true. If it’s “too far,” you reject the hypothesis; otherwise, you fail to reject it. Now maybe the hypothesized mean is correct, and you just happened to get “bad” data (an extreme sample). In fact, that can happen, and classical hypothesis testing allows for that possibility. If your sample is unrepresentative (just by random chance or by poor sampling), you could see an extreme sample mean that leads you to reject the hypothesis. Classical hypothesis testing tries to control how often that happens (the Type I error rate) across many repeated experiments.

In classical hypothesis testing (e.g., a z‐test or t‐test), the “hypothesized mean” is specifically the population mean you assume (under the null hypothesis) before you look at the sample. You’re effectively saying: $H_0: \mu = \mu_0$ where $\mu_0$ is the hypothesized population mean. Then, using your sample mean $\bar{x}$, you test whether the data are so far from $\mu_0$ as to be “unlikely” under that assumption. The key point is: we usually don’t actually know the true population mean in real-life situations. We only have:
- A claim (e.g., “the manufacturer says the mean battery life is 100 hours”), or
- A theory (e.g., “historical data suggest the mean is 50”), or
- Some guess based on partial information.
That hypothesized value $\mu_0$ is what we call the population mean in the null hypothesis. But in practice, it might be the manufacturer’s claim, a researcher’s theory, or some other reference point. We test it against our sample to see if our data conflict with that assumption.

**If you already know the true population mean (i.e., you literally have data for the entire population, or there is some proven fact about that mean), then there’s no need to run a z‐test or t‐test to “check” it. Why? Because the purpose of these tests is to infer or test a hypothesis about the population mean from a sample. If the population mean is truly known and undisputed, there is no uncertainty to resolve. In other words, hypothesis testing is only necessary when the parameter (like a mean) is unknown and we use sample data to draw conclusions about it.**

We now need to clear up the concept of a population standard deviation. A hypothesized population mean $\mu_0$ is simply the value you’re testing in your null hypothesis: “I assume the true mean is 100” (for instance), and I want to see if my sample data conflict with that assumption. A population standard deviation σ is about the spread (variability) in the population. It can come from completely different sources—large historical data, industry/engineering specs, or repeated past measurements that have established how much the data typically vary. This means: Even if you aren’t sure the true mean is 100 (that’s exactly what you’re testing!), you could still have high confidence that the standard deviation is about 10, because many previous studies or manufacturing records have shown the process always has around 10 hours of variability. Hence, in a z‐test:
- You bring a hypothesized $\mu_0$ (the population mean you want to test).
- You also bring a known population standard deviation 𝜎 (from historical/industrial knowledge) that describes how wide the distribution is around whatever the true mean is.
They don’t have to come from the same source, and one doesn’t depend on the other. You’re simply saying, “Given we know the standard deviation is about 10 (from large past data), let’s see if new sample data conflict with the claim that $H_0: \mu = \mu_0$.”

In classical stats testing, you may have a population mean (hypothesized mean), a population standard deviation and your own sample mean based on sample data. Why do you need a sample mean in z-tests? You need the sample mean in a z‐test because the whole purpose of the test is to see how far your observed average (from your sample) deviates from the hypothesized population mean $\mu_0$. In other words:
- Null Hypothesis: $H_0: \mu = \mu_0$
    - You propose a value $\mu_0$ for the true mean of the population.
- Observed Sample Mean $\bar{x}$
    - You collect data (a sample) and calculate $\bar{x}$. This is your best estimate of the actual (but unknown) population mean $\mu$.
- Test Statistic z:
    - $z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$
        - σ is the known population standard deviation (from external info).
        - 𝑛 is the sample size.
        - The numerator $\bar{x} - \mu_0$ measures how far your sample is from the hypothesized mean.
    - A test statistic 𝑧 is essentially a standardized measure of how far your observed data (e.g., a sample mean) deviates from the hypothesis measured in units of the standard error. Is it measured in units of standard error o z-scores? They’re effectively the same concept in this context. A z-score is precisely a measurement of “how many standard deviations (or standard errors) above/below the mean” an observed value is. Hence, for a z-test, being measured in units of the standard error is the same thing as being a z-score.
- Interpretation:
    - If $\bar{x}$ is very far from $\mu_0$ relative to $\sigma / \sqrt{n}$,  then the z‐value will be large in magnitude, meaning the sample mean is “unusually far” from $\mu_0$ (assuming $H_0$ were true). You might then reject $H_0$.
    - If $\bar{x}$ is close to $\mu_0$, the test statistic will be small, suggesting no strong evidence that the true mean differs from $\mu_0$.

How 𝑧 Relates to the Standard Normal Distribution

A normal distribution is any bell-shaped probability distribution that can be completely described by its mean μ and standard deviation σ. In mathematical form, if a random variable X is normally distributed, its probability density function is:
$$
f_X(x) = \frac{1}{\sigma \sqrt{2\pi}} 
\, \exp\!\Bigl(-\frac{(x - \mu)^2}{2\sigma^2}\Bigr).
$$
The normal distribution’s probability density function (PDF) can be understood as follows:
- Center at the Mean: The highest point of the curve is at x=μ. This reflects that values closest to the mean are the most likely to occur.
- Symmetry Around the Mean: The expression $(x - \mu)^2$ inside the exponential penalizes deviations from the mean equally on both sides (remember squaring a number distributes it equally on both sides forming a parabola). The distribution is symmetric about its mean.
- Spread Controlled by 𝜎: The term 𝜎 (the standard deviation) appears in both the denominator in front and inside the exponent. A larger σ results in a “wider” curve (since deviations from the mean are less penalized), while a smaller 𝜎 creates a “narrower” curve (deviations from the mean become more sharply penalized).
- Exponential Decay: The exponential $\exp\!\Bigl(-\frac{(x - \mu)^2}{2\sigma^2}\Bigr)$ ensures that the probability density decreases very quickly as x moves away from μ. Farther values from μ become less probable.
    - Exponential Decay in plain English: Imagine you have a hill centered at 𝜇. Right at μ, the hill is at its highest point. As you move away from 𝜇, the “height” (which represents the probability density) starts to drop. Because there’s an exponential term, that drop happens very fast: a small increase in (x−μ) leads to a fairly big negative increase inside the exponent, causing the entire expression to shrink dramatically. As a result, values of 𝑥 far from 𝜇 end up with a very small probability density. Essentially, that exponential factor is what gives the normal distribution its characteristic bell shape, ensuring most of the probability mass stays near 𝜇, while the tails drop off sharply. 
- Normalization by: $\frac{1}{\sigma \sqrt{2\pi}} $: This term ensures that the total area under the curve is 1. 
    - The constant $\sqrt{2\pi}$ appears in many continuous probability distributions.
    - Dividing by σ scales the curve appropriately so that the integral over all real values of 𝑥 equals 1.
The normal distribution can be shown as a smooth, bell-shaped curve on a graph, where the horizontal axis is the value 𝑥 and the vertical axis is the probability density for that 𝑥.
- The total area under this bell curve is 1, representing 100% of all possible outcomes.
- The area between any two points on the 𝑥-axis (say between 𝑎 and 𝑏) corresponds to the probability that the random variable takes on a value within that interval [a,b]. 
In other words, even though the height of the curve at a specific point 𝑥 is not the probability of 𝑥 exactly, the area under the curve over a range gives you the probability of being in that range. This way, the normal distribution curve provides an intuitive picture of where the random variable is “most likely” to fall (around 𝜇) and how quickly likelihood tapers off for values far from 𝜇. 

Why are most things in nature normally distributed? Many random processes in nature tend to be influenced by a large number of small, independent factors, each of which adds or subtracts a little from the outcome. By the Central Limit Theorem (CLT), when you sum (or average) enough of these independent random “contributions,” the result tends to approximate a normal distribution, regardless of the underlying distribution of the individual factors.

In simpler terms:
- Think of height, for example. It’s influenced by hundreds of small genetic and environmental factors. No single factor dominates the outcome; instead, you get many tiny pushes in different directions.
- The CLT implies that when you add up all these little effects, the distribution of their sum (or average) often becomes bell-shaped.
- As a result, many measurements or traits in nature—like heights, test scores, measurement errors—are often found to be roughly normal in real data.

The features of the normal distribution formula work together to create the familiar “bell shape.” Because of the squared term in the exponent, the distribution prioritizes values near μ, while making values far from μ increasingly unlikely, providing a smooth, continuous shape that has no sharp edges or discontinuities.

The standard normal distribution is the special case of the normal distribution with mean μ=0 and standard deviation σ=1. The formula is represented as so:
$$
Z = \frac{X - \mu}{\sigma}
$$
- X is a random variable that follows a normal distribution with mean 𝜇 and standard deviation 𝜎. For example, suppose 𝑋 s the random variable representing the height (in inches) of an adult man in a certain population. We might model 𝑋 as approximately normally distributed with mean 𝜇 = 70 inches and standard deviation 𝜎 = 3 inches. Each individual man in that population has a specific height, but if you randomly select one individual, their height is uncertain—hence, it’s a random variable.
- Z is then the standardized version of 𝑋, which follows a standard normal distribution (mean 0 and standard deviation 1). **When we talk about standardizing, we’re not literally changing your physical height. Instead, we’re re-expressing your height in terms of how many standard deviations it is above or below the average within a certain population. If your height is 𝑋=73 inches, and the population mean is 𝜇 = 70, and the standard deviation is 𝜎 =3, then:** 
$$
Z = \frac{73 - 70}{3} = 1
$$
This just means your height is 1 standard deviation above the average. The standardization formula converts a real-world measurement (your height) into a unitless score (the 𝑍-score). It tells us where you stand relative to everyone else, in terms of standard deviations.

**Any normal distribution can be converted into a standard normal distribution by “standardizing”**: $Z = \frac{X - \mu}{\sigma}$. 

The standard normal distribution and its z-scores are extremely important for understanding probabilities in normally distributed data. Here’s why:
- Universal Reference Point
    - A z-score (also called a “standard score”) tells you how many standard deviations an observation is above or below the mean.
    - By standardizing different normal variables, you reduce them to the same “unit,” allowing direct comparisons across different measurements and scales.
- Simplified Probability Lookups
    - **Instead of having separate probability tables (or complicated integrals) for every combination of mean 𝜇 and standard deviation 𝜎, you can look up (or calculate) probabilities from a single standard normal table (or function) once you convert your data to z-scores.**
- Foundation for Many Statistical Methods
    - Confidence intervals, hypothesis tests (like z-tests), and other inferential procedures often rely on the assumption that a test statistic follows a (standard) normal distribution when sample sizes are large or under ideal conditions.
    - Mastering z-scores and the standard normal distribution paves the way to understanding these more advanced techniques.
Overall, z-scores capture the essence of “how far and in what direction an observation is from the mean, measured in standard deviations,” and this unifying perspective underlies much of classical statistics and probability theory.

Standard Normal Distribution and Cumulative Distribution Function (CDF)

As we learned in descriptive statistics, a cumulative distribution function (CDF) for a random variable 𝑋 is a function that, for any number 𝑥, tells you the probability that 𝑋 is less than or equal to 𝑥. Mathematically, it’s written as: 
$$
F_X(x) = P\bigl(X \le x\bigr)
$$
Here’s what that means in simpler terms:
- When you plug a value 𝑥 into the CDF, you get the fraction of all possible outcomes that lie up to (and including) 𝑥.
- Because $F_X(x)$ represents a probability, it is always between 0 and 1.
- As 𝑥 increases from −∞ to +∞, $F_X(x)$ grows (or at least doesn’t decrease) from 0 up to 1.

Suppose 𝑋 is the random variable representing the height (in inches) of an adult woman in a certain population. Assume 𝑋 is (approximately) normally distributed with a mean of 𝜇=64 inches and a standard deviation of 𝜎=2.5 inches: $X \sim \mathcal{N}(64, 2.5^2)$.

The cumulative distribution function (CDF) of 
𝑋, denoted by $F_X(x)$, gives the probability that a randomly selected adult woman in this population has a height less than or equal to 𝑥. Symbolically,
$$
F_X(x) = P\bigl(X \le x\bigr)
$$

If you wanted to know what proportion of women in this population are 5'4" (64 inches) or shorter, you would compute:
$$
F_X(64) = P\bigl(X \le 64\bigr)
$$

Because $F_X(x)$ is the area under the normal bell curve from −∞ to 𝑥, it fully describes the distribution—meaning you can use it to find probabilities for any range of heights.
 
Why is CDF useful for z tables? A z-table is essentially a lookup for the cumulative distribution function (CDF) of the standard normal distribution.  In other words, a z-table directly tells you $P(Z \le z)$ for a standard normal variable 𝑍. So, if you want to find the probability that 𝑍 is less than or equal to some value 𝑧, you look up the CDF in the table. **Because the CDF tells us the area under the curve up to a point, it perfectly aligns with what a z-table provides: the proportion (or area) of a standard normal distribution from −∞ up to 𝑧. This is why the CDF concept is at the heart of how z-tables work.** 

Standard Normal Distribution and P-Value

Is the CDF and p-value the same thing in standard normal distribution? No, they’re not the same thing, though they are closely related:
- The CDF of a standard normal random variable $Z \sim \mathcal{N}(0,1)$ at a point 𝑧 is the probability that 𝑍 is less than or equal to 𝑧. Symbolically:
$$
F_Z(z) = P(Z \le z)
$$
- A p-value for a test (like a z-test) is the probability of seeing a test statistic at least as extreme as the observed one under the null hypothesis. For a two-sided z-test, for instance:
$$
\text{p-value} = 2 \times P\bigl(Z \ge \lvert z_{\text{observed}} \rvert\bigr)
$$
where $z_{\text{observed}}$ is the actual computed z-statistic from your data. Remember in a z-test, the observed z-statistic $z_{\text{observed}}$ is computed by taking the difference between the sample mean $\bar{x}$ and the hypothesized population mean $\mu_0$, then dividing by the standard error (which, if 𝜎 is known), is $\frac{\sigma}{\sqrt{n}}$. Symbolically:
$$
z_{\text{observed}}  
= \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$
**Essentially, the p-value calculation makes use of the CDF (or complementary CDF) to measure how unlikely (under the null hypothesis) the observed result (or something more extreme) is. But the raw CDF simply gives “the probability up to this point,” not specifically the probability of being at least as extreme in both tails.**

  

 
- Intuition: Central Position and Unit Spread
    - Because it’s centered at 0, we measure deviations as positive or negative (above or below the mean).
    - Having a standard deviation of 1 means these deviations are in terms of “standard units,” so it’s straightforward to interpret a value like Z=1.96 as being about “1.96 standard deviations above the mean.”

With regards to the test statistic z in a z-test, once you compute the test statistic 𝑧, you can think of it as a point on the horizontal axis of this standard normal curve.
    - If 𝑧=0, it sits at the center (mean = 0)
    - If 𝑧 is positive, it’s to the right of center; negative 𝑧 is to the left. The magnitude $\lvert z \rvert$ tells you how many standard deviations away from 0 you are.

Decide on the Significance Level 𝛼

The choice of 𝛼 = 0.05 (a 5% significance level) is primarily conventional — it arose historically in statistics through the influential work of Ronald A. Fisher and others. Over time, it became a common rule of thumb for distinguishing results that are “statistically significant” from those that are not. 

It helps balancing Type I and Type II errors. A Type I error is rejecting the null hypothesis when it is actually true. 𝛼 is the maximum risk of committing a Type I error that you’re willing to accept. **At 𝛼 = 0.05, you’re accepting up to a 5% chance of concluding there’s an effect (or difference) when in fact there isn’t.**  Is this because the confidence interval is 95 percent and when it falls inside the 95 percent confidence interval we fail to reject the null hypothesis?

Yes—that 5% significance level (𝛼 = 0.05) and the 95% confidence interval are directly connected in the usual two-sided test. Here’s how:
- If the hypothesized value ($\mu_0$, for example)  lies outside the corresponding 95% confidence interval, then you reject $H_0$ at the 5% level.
- If that hypothesized value lies inside the 95% confidence interval, you fail to reject  $H_0$ at the 5% level.
In other words, a 95% confidence interval can be viewed as all the parameter values that would not be rejected by a two-sided test with α=0.05. Under the hood, setting α=0.05 means you accept up to a 5% chance (in repeated sampling) of rejecting the null hypothesis when it is actually true. A 95% confidence interval means that, in repeated sampling, 95% of those confidence intervals will contain the true parameter. These two perspectives line up so that “95% confident” corresponds precisely to “at most 5% chance of a false positive.” 

How to Decide Whether to Reject or Fail to Reject $H_0$

We now have a z-test test statistic. We have a signifance level. The next steps typically go like this to reject or fail to reject the null hypothesis:

Method A: Using a p-value
p-value definition: The p-value is the probability, under the assumption that the null hypothesis ($\mu = \mu_0$) is true, of getting a z-value as extreme or more extreme than the one you got.

Finding the p-value:
- If you do a two-sided test (aka two-tailed test),  (i.e., you care about deviations in either direction), it means we’re checking if the sample mean is either significantly above or significantly below the hypothesized mean. Hence, “extremes” lie in both ends (“tails”) of the distribution. Here’s how to see it step by step:



Method B: Using Critical Values

Using a standard normal distribution, you can define a confidence interval built around your sample mean bound by two z-scores defining the curve if it is a two-tail test. You can run a z-test to check if the hypothesized value lies within that interval. If it does, you generally fail to reject the hypothesis at the chosen significance level. This tells you whether your observed data (sample mean) is statistically consistent (not too far off) with the hypothesized mean, given a certain cutoff (e.g. 5% significance). We don’t strictly say “accurate” in the sense of “true or false”; rather, we say whether or not the difference is statistically significant. This is in essence a z-test or when you don't have the population standard deviation and you have a sample standard deviation, then a t-test, which uses the degree of freedom principle.

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