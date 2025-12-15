module WilksLambdaGroupedDFHandler

    using Statistics, MultivariateStats, Plots, Random, DataFrames, LinearAlgebra

    export obtain_variable_matrix_from_different_samples, obtain_means_for_variable_in_each_sample, obtain_E_and_H_matrix_for_wilks_lambda

    #The assumption is that each sample is a DataFrame with the same variable columns and same sample size
    function obtain_variable_matrix_from_different_samples(df::Vector{DataFrame}, sample_size::Int)
        #Get variables that are important
        variable_names = names(df[1])[1:end-1]
        num_variables = length(variable_names)
        number_of_samples = length(df)
        matrix_of_different_variables = Matrix{Matrix{Float64}}(undef,1, num_variables)

        for (i,name) in enumerate(variable_names)
            variable_matrix = Matrix{Float64}(undef, sample_size, number_of_samples)
            for (j,sample_df) in enumerate(df)
                variable_matrix[:, j] = sample_df[:, name][:,:]
            end
            matrix_of_different_variables[1, i] = variable_matrix
        end
        return matrix_of_different_variables, variable_names
    end

    function obtain_means_for_variable_in_each_sample(dfvec::Vector{DataFrame})
        num_samples = length(dfvec)

        means_matrix = Matrix{Float64}(undef, num_samples, length(names(dfvec[1])) - 1)

        for i in 1:num_samples
            means_matrix[i, :] = [mean(dfvec[i][!, name]) for name in names(dfvec[i])[1:end-1]]
        end
        return means_matrix
    end

    function obtain_E_and_H_matrix_for_wilks_lambda(matrix_of_variable_matrices::Matrix{Matrix{Float64}}, means_matrix::Matrix{Float64})
        num_variables = length(matrix_of_variable_matrices)
        num_samples = size(matrix_of_variable_matrices[1], 2)
        E_matrix = zeros(num_variables, num_variables)
        H_matrix = zeros(num_variables, num_variables)
        overall_mean_vector = mean(means_matrix, dims=1) |> Matrix{Float64}

        #Center each sample matrix
        centered_matrices = Matrix{Matrix{Float64}}(undef, 1, length(matrix_of_variable_matrices))
        for i in 1:length(matrix_of_variable_matrices)
            centered_matrices[1, i] = Matrix{Float64}(undef, size(matrix_of_variable_matrices[1, i], 1), num_samples)
            for j in 1:num_samples
                centered_matrices[1, i][:, j] = matrix_of_variable_matrices[1, i][:, j] .- means_matrix[j, i]
            end
        end

        #Compute E matrix
        for i in 1:num_variables
            for j in 1:num_variables
                for k in 1:num_samples
                    E_matrix[i,j] += dot(centered_matrices[1, i][:, k], centered_matrices[1, j][:, k])
                end
            end
        end
        #Compute H matrix
        for i in 1:num_variables
            for j in 1:num_variables
                for k in 1:num_samples
                    mean_diff_i = means_matrix[k, i] - overall_mean_vector[1, i]
                    mean_diff_j = means_matrix[k, j] - overall_mean_vector[1, j]
                    H_matrix[i,j] += mean_diff_i * mean_diff_j * size(matrix_of_variable_matrices[1, i], 1)
                end
            end
        end

        println(H_matrix)

        return E_matrix, H_matrix
    end

end