"""
    Ch187Backprop

Backpropagation from scratch for a feedforward neural network (Julia).

Mirrors the Python module `aiinaction.ch187_backprop` and the Rust module
`aiinaction::ch187_backprop`. The shared fixtures in `test/test_ch187_backprop.jl`
match the Python/Rust suites, which keeps the three implementations at parity.

The network is a plain multilayer perceptron: each hidden layer applies the
logistic sigmoid elementwise, the output layer is linear (identity), and the loss
is one-half squared error `C = 1/2 ||a^L - y||^2`. Weights of layer `l` form a
matrix `W^l` of shape `(n_l, n_{l-1})` stored as `Matrix{Float64}`.

Equations: BP1 `delta^L = a^L - y` (linear output); BP2
`delta^l = ((W^{l+1})' * delta^{l+1}) .* sigma'(z^l)`; BP3 `dC/db^l = delta^l`;
BP4 `dC/dW^l = delta^l * (a^{l-1})'`.
"""
module Ch187Backprop

export MLP, make_mlp, sigmoid, sigmoid_prime, forward, squared_error_loss,
    backprop, numerical_gradient

"""A feedforward network with sigmoid hidden layers and a linear output."""
struct MLP
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
end

num_layers(net::MLP) = length(net.weights)
n_input(net::MLP) = size(net.weights[1], 2)
n_output(net::MLP) = size(net.weights[end], 1)

"""
    make_mlp(weights, biases)

Validate layer shapes and finiteness, returning an `MLP`. Each `weights[l]` has
shape `(n_{l+1}, n_l)` and `biases[l]` length `n_{l+1}`.
"""
function make_mlp(weights, biases)
    length(weights) == length(biases) ||
        throw(ArgumentError("weights and biases must have equal length, got $(length(weights)) and $(length(biases))"))
    isempty(weights) && throw(ArgumentError("network must have at least one layer"))

    W = Vector{Matrix{Float64}}(undef, length(weights))
    b = Vector{Vector{Float64}}(undef, length(biases))
    prev_cols = nothing
    for l in eachindex(weights)
        wl = Matrix{Float64}(weights[l])
        bl = Vector{Float64}(biases[l])
        size(wl, 1) == length(bl) ||
            throw(ArgumentError("layer $(l - 1): weights has $(size(wl, 1)) rows but bias has length $(length(bl))"))
        if prev_cols !== nothing && size(wl, 2) != prev_cols
            throw(ArgumentError("layer $(l - 1): weights expects $(size(wl, 2)) inputs but previous layer emits $prev_cols"))
        end
        (all(isfinite, wl) && all(isfinite, bl)) ||
            throw(ArgumentError("layer $(l - 1): parameters contain non-finite values (nan or inf)"))
        prev_cols = size(wl, 1)
        W[l] = wl
        b[l] = bl
    end
    return MLP(W, b)
end

"""Numerically stable logistic sigmoid."""
function sigmoid(z::Real)
    if z >= 0
        return 1.0 / (1.0 + exp(-z))
    else
        ez = exp(z)
        return ez / (1.0 + ez)
    end
end
sigmoid(z::AbstractVector) = sigmoid.(z)

"""Derivative `sigma'(z) = sigma(z) (1 - sigma(z))`."""
function sigmoid_prime(z::Real)
    s = sigmoid(z)
    return s * (1.0 - s)
end
sigmoid_prime(z::AbstractVector) = sigmoid_prime.(z)

"""
    forward(net, x) -> (zs, activations)

Run the forward pass. `zs[l]` is the pre-activation `z^{l+1}`, `activations[l]`
is `a^l` with `activations[1] == x`. Hidden layers use the sigmoid, the output is
linear.
"""
function forward(net::MLP, x)
    a = Vector{Float64}(x)
    length(a) == n_input(net) ||
        throw(ArgumentError("x has length $(length(a)) but network expects $(n_input(net))"))
    all(isfinite, a) || throw(ArgumentError("x contains non-finite values (nan or inf)"))

    L = num_layers(net)
    activations = Vector{Vector{Float64}}(undef, L + 1)
    zs = Vector{Vector{Float64}}(undef, L)
    activations[1] = a
    for l in 1:L
        z = net.weights[l] * a + net.biases[l]
        a = l == L ? z : sigmoid(z)
        zs[l] = z
        activations[l + 1] = a
    end
    return zs, activations
end

"""One-half squared-error loss `1/2 ||a^L - y||^2` for one example."""
function squared_error_loss(net::MLP, x, y)
    yv = Vector{Float64}(y)
    length(yv) == n_output(net) ||
        throw(ArgumentError("y has length $(length(yv)) but network outputs $(n_output(net))"))
    _, activations = forward(net, x)
    diff = activations[end] .- yv
    return 0.5 * sum(abs2, diff)
end

"""
    backprop(net, x, y) -> (grad_W, grad_b)

Compute parameter gradients for one example via backpropagation. `grad_W[l]`
matches `net.weights[l]` and `grad_b[l]` matches `net.biases[l]`.
"""
function backprop(net::MLP, x, y)
    yv = Vector{Float64}(y)
    length(yv) == n_output(net) ||
        throw(ArgumentError("y has length $(length(yv)) but network outputs $(n_output(net))"))

    zs, activations = forward(net, x)
    L = num_layers(net)
    grad_W = [zeros(Float64, size(w)) for w in net.weights]
    grad_b = [zeros(Float64, length(b)) for b in net.biases]

    # BP1: linear output, delta^L = a^L - y.
    delta = activations[end] .- yv
    grad_b[L] = delta                            # BP3
    grad_W[L] = delta * activations[L]'          # BP4

    # BP2: propagate backward through the hidden (sigmoid) layers.
    for l in (L - 1):-1:1
        delta = (net.weights[l + 1]' * delta) .* sigmoid_prime(zs[l])
        grad_b[l] = delta                        # BP3
        grad_W[l] = delta * activations[l]'      # BP4
    end

    return grad_W, grad_b
end

"""
    numerical_gradient(net, x, y; eps=1e-6) -> (grad_W, grad_b)

Central-difference estimate of the loss gradient, for gradient checking.
"""
function numerical_gradient(net::MLP, x, y; eps::Float64=1e-6)
    eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
    grad_W = [zeros(Float64, size(w)) for w in net.weights]
    grad_b = [zeros(Float64, length(b)) for b in net.biases]

    for l in 1:num_layers(net)
        w = net.weights[l]
        for j in 1:size(w, 1), k in 1:size(w, 2)
            orig = w[j, k]
            w[j, k] = orig + eps
            cp = squared_error_loss(net, x, y)
            w[j, k] = orig - eps
            cm = squared_error_loss(net, x, y)
            w[j, k] = orig
            grad_W[l][j, k] = (cp - cm) / (2 * eps)
        end
        b = net.biases[l]
        for j in 1:length(b)
            orig = b[j]
            b[j] = orig + eps
            cp = squared_error_loss(net, x, y)
            b[j] = orig - eps
            cm = squared_error_loss(net, x, y)
            b[j] = orig
            grad_b[l][j] = (cp - cm) / (2 * eps)
        end
    end
    return grad_W, grad_b
end

end # module Ch187Backprop
