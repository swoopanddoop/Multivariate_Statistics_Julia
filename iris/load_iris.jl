using CSV, DataFrames, Statistics, MultivariateStats, Plots, Random, LinearAlgebra

include("./wilks_lambda_grouped_df_handler.jl")
using .WilksLambdaGroupedDFHandler

#Load the dataset
df = CSV.read("./iris.csv", DataFrame)

df_grouped = groupby(df, :variety)

Random.seed!(123)
sampled_rand = vcat([g[randperm(nrow(g))[1:50], :] for g in df_grouped if nrow(g) >= 50])

println("Sampled DataFrame shape: ", size(sampled_rand[1]))

matrix_of_different_variables, variable_names = obtain_variable_matrix_from_different_samples(sampled_rand, 50)

println("Shape of variable matrix: ", size(matrix_of_different_variables[1]))

means_matrix = obtain_means_for_variable_in_each_sample(sampled_rand)

println("Means matrix shape: ", size(means_matrix))

E_matrix, H_matrix = obtain_E_and_H_matrix_for_wilks_lambda(matrix_of_different_variables, means_matrix)

wilks_lambda = det(E_matrix) / det(E_matrix + H_matrix)
println("Wilk's Lambda: ", wilks_lambda)
