module TruncatedPopulations

using Distributions
using Random
using Random: default_rng
using Statistics
using StatsFuns
using Turing

export scotts_rule_bins
export draw_truncated_population, draw_observed_population, draw_likelihood_samples
export exact_likelihood_model, samples_model

raw"""
    scotts_rule_bins(data)

Returns the number of histogram bins to fit `data` assuming the bin with is set
by Scott's rule.

Scott's rule suggests a bin width of 

``h = 3.5 * \sigma / n^{1/3}``

where ``\sigma`` is the standard deviation of the data and ``n`` is the number
of data points; this is based on the bin width that will minimize the expected
integrated squared error of the histogram as an estimate of the underlying
normal distribution.  The method here replaces ``\sigma`` with half the `0.16`
and `0.84` quantiles of the data, which is a robust measure of the spread of the
data that is less sensitive to outliers than the standard deviation.
"""
function scotts_rule_bins(data)
    sigma = (quantile(data, 0.84) - quantile(data, 0.16))/2

    h_scott = 3.5 * sigma / length(data)^(1/3)

    n_bins = ceil(Int, (maximum(data) - minimum(data)) / h_scott)

    n_bins
end

"""
    draw_truncated_population(mu_true, sigma_true, n; lower=0.0, upper=1.0)

Draw `n` samples from a truncated normal distribution with mean `mu_true`,
standard deviation `sigma_true`, and truncation limits `lower` and `upper`.
"""
function draw_truncated_population(mu_true, sigma_true, n; lower=0.0, upper=1.0, rng=default_rng())
    rand(rng, truncated(Normal(mu_true, sigma_true), lower, upper), n)
end

"""
    draw_observed_population(qs_true, sigma_obs)

Draw observed population values by adding normally distributed noise with
standard deviation `sigma_obs` (assumed to be a scalar) to the true population
values `qs_true`. The resulting observed population values are returned as a
vector of the same length as `qs_true`.
"""
function draw_observed_population(qs_true, sigma_obs; rng=default_rng())
    [rand(rng, Normal(q, sigma_obs)) for q in qs_true]
end

"""
    exact_likelihood_model(qs_obs, sigma_obs; lower=0.0, upper=1.0)

Defines a Turing model for the exact likelihood of observed population values
`qs_obs` given a truncated normal distribution for the true population values.
The model includes parameters `mu` and `sigma` for the mean and standard
deviation of the truncated normal distribution, and it models the observed
population values as normally distributed around the true population values with
standard deviation `sigma_obs`. The truncation limits for the true population
values are specified by `lower` and `upper`.  The model samples over the true
population values using a non-centered parameterization, which can improve
sampling efficiency and convergence in hierarchical models *when the
observational uncertainty is comparable to or larger than the population width*.
"""
@model function exact_likelihood_model(qs_obs, sigma_obs; lower=0.0, upper=1.0)
    mu ~ Normal(0.0, 1.0)
    sigma ~ truncated(Normal(0.0, 1.0), 0.0, Inf)

    qs_raw = Vector{Float64}(undef, length(qs_obs))
    for i in eachindex(qs_raw)
        qs_raw[i] ~ truncated(Normal(0, 1), (lower-mu) / sigma, (upper-mu) / sigma)
    end
    qs := mu .+ sigma .* qs_raw

    for i in eachindex(qs_obs)
        qs_obs[i] ~ Normal(qs[i], sigma_obs)
    end
end

"""
    draw_likelihood_samples(qs_obs, sigma_obs; lower=0.0, upper=1.0, n_samples=1000)

Draw samples from the likelihood of observed population values `qs_obs` assuming
additive Gaussian noise with standard deviation `sigma_obs`.  Samples are
returned as a matrix `(n_samples, length(qs_obs))` where each column corresponds
to a sample of the true population values from the likelihood for the
corresponding observed population value.  The samples are truncated to be within
the limits specified by `lower` and `upper`.
"""
function draw_likelihood_samples(qs_obs, sigma_obs; lower=0.0, upper=1.0, n_samples=1000, rng=default_rng())
    samps = [rand(rng, truncated(Normal(qo, sigma_obs), lower, upper), n_samples) for qo in qs_obs]
    stack(samps, dims=2)
end

"""
    samples_model(q_samples; lower=0.0, upper=1.0)

Defines a Turing model for the truncated normal population given samples from
drawn from the likelihood of observed population values. The model includes
parameters `mu` and `sigma` for the mean and standard deviation of the truncated
normal distribution, and it generates the effective number of samples (`neff`)
for the Monte-Carlo likelihood marginalization for each observed population
value based on the likelihood samples.
"""
@model function samples_model(q_samples; lower=0.0, upper=1.0)
    mu ~ Normal(0.0, 1.0)
    sigma ~ truncated(Normal(0.0, 1.0), 0.0, Inf)

    logps = logpdf.(Ref(truncated(Normal(mu, sigma), lower, upper)), q_samples)

    marginal_logls = logsumexp(logps, dims=1)

    log_neff = vec(2.0 .* marginal_logls .- logsumexp(logps .* 2, dims=1))
    neff := exp.(log_neff)

    Turing.@addlogprob! sum(marginal_logls)
end

end # module TruncatedPopulations
