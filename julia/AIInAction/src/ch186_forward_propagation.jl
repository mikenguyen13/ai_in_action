"""
    Ch186ForwardPropagation

Feedforward neural network forward propagation from scratch (Julia).

Mirrors the Python module `aiinaction.ch186_forward_propagation` and the Rust
module `aiinaction::ch186_forward_propagation`. The shared fixtures in
`test/test_ch186_forward_propagation.jl` match the Python/Rust suites, which keeps
the three implementations at parity.

A batch is a `d x m` matrix (features down rows, examples across columns), matching
the column-batch convention used in the chapter. Each layer computes
`Z = W * A_prev .+ b` and applies an elementwise activation. The sigmoid is
evaluated in a numerically stable, branch-free form so it never overflows.
"""
module Ch186ForwardPropagation

export Layer, make_layer, forward_layer, forward,
    relu, sigmoid, tanh_act, identity_act, apply_activation, ACTIVATIONS

const ACTIVATIONS = ("relu", "sigmoid", "tanh", "identity")

"""Rectified linear unit, `max(0, z)`, applied elementwise."""
relu(z) = max.(z, 0.0)

"""Numerically stable logistic sigmoid `1 / (1 + e^{-z})`, applied elementwise."""
function sigmoid(z)
    return map(z) do x
        if x >= 0
            1.0 / (1.0 + exp(-x))
        else
            ex = exp(x)
            ex / (1.0 + ex)
        end
    end
end

"""Hyperbolic tangent, applied elementwise."""
tanh_act(z) = tanh.(z)

"""Identity (linear) activation, returns `z` unchanged."""
identity_act(z) = z

"""Dispatch to the named activation, erroring on unknown names."""
function apply_activation(name::AbstractString, z)
    if name == "relu"
        return relu(z)
    elseif name == "sigmoid"
        return sigmoid(z)
    elseif name == "tanh"
        return tanh_act(z)
    elseif name == "identity"
        return identity_act(z)
    else
        throw(ArgumentError("unknown activation \"$name\"; expected one of $(ACTIVATIONS)"))
    end
end

"""One dense layer: an affine map `W * a .+ b` plus an activation.

`W` has shape `(d_out, d_in)` and `b` has length `d_out`."""
struct Layer
    W::Matrix{Float64}
    b::Vector{Float64}
    activation::String
end

d_in(l::Layer) = size(l.W, 2)
d_out(l::Layer) = size(l.W, 1)

"""
    make_layer(W, b, activation)

Validate and construct a `Layer` from raw weights, bias and activation name.
"""
function make_layer(W, b, activation::AbstractString)
    Wm = W isa AbstractVector ? reshape(Float64.(W), 1, length(W)) : Matrix{Float64}(W)
    bv = Vector{Float64}(b)
    activation in ACTIVATIONS ||
        throw(ArgumentError("unknown activation \"$activation\"; expected one of $(ACTIVATIONS)"))
    length(bv) == size(Wm, 1) ||
        throw(ArgumentError("bias length $(length(bv)) does not match number of units $(size(Wm, 1))"))
    all(isfinite, Wm) || throw(ArgumentError("W contains non-finite values (nan or inf)"))
    all(isfinite, bv) || throw(ArgumentError("b contains non-finite values (nan or inf)"))
    return Layer(Wm, bv, String(activation))
end

"""
    forward_layer(layer, A_prev) -> (Z, A)

Propagate one batched activation through a single layer. `Z = W * A_prev .+ b`
(bias broadcast across columns) and `A` is the activation of `Z`; both are
`d_out x m`. `Z` is returned alongside `A` because it is what a backward pass caches.
"""
function forward_layer(layer::Layer, A_prev::AbstractMatrix)
    size(A_prev, 1) == d_in(layer) ||
        throw(ArgumentError("layer expects $(d_in(layer)) input features but received $(size(A_prev, 1))"))
    Z = layer.W * A_prev .+ layer.b
    A = apply_activation(layer.activation, Z)
    return Z, A
end

_as_batch(x::AbstractVector) = reshape(Float64.(x), :, 1)
_as_batch(x::AbstractMatrix) = Matrix{Float64}(x)

"""
    forward(layers, X) -> A_L

Run the full forward sweep through every layer in order. `X` is a `d0 x m` batch
with examples as columns; a length-`d0` vector is treated as a single example.
Returns the network output `A^[L]` of shape `(d_L, m)`.
"""
function forward(layers::AbstractVector{Layer}, X)
    isempty(layers) && throw(ArgumentError("network must have at least one layer"))
    for i in 1:(length(layers) - 1)
        d_out(layers[i]) == d_in(layers[i + 1]) ||
            throw(ArgumentError("layer $(i - 1) outputs $(d_out(layers[i])) features but layer $i expects $(d_in(layers[i + 1]))"))
    end
    A = _as_batch(X)
    size(A, 1) == d_in(layers[1]) ||
        throw(ArgumentError("input has $(size(A, 1)) features but first layer expects $(d_in(layers[1]))"))
    all(isfinite, A) || throw(ArgumentError("input contains non-finite values (nan or inf)"))
    for layer in layers
        _, A = forward_layer(layer, A)
    end
    return A
end

end # module Ch186ForwardPropagation
