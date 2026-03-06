module TruncatedPopulations

export scotts_rule_bins

using Statistics

function scotts_rule_bins(data)
    sigma = (quantile(data, 0.84) - quantile(data, 0.16))/2

    h_scott = 3.5 * sigma / length(data)^(1/3)

    n_bins = ceil(Int, (maximum(data) - minimum(data)) / h_scott)

    n_bins
end

end # module TruncatedPopulations
