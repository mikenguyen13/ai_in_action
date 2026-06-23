"""
    Ch198LrSchedules

Learning rate schedules from scratch (Julia).

Mirrors the Python module `aiinaction.ch198_lr_schedules` and the Rust module
`aiinaction::ch198_lr_schedules`. The shared fixtures in
`test/test_ch198_lr_schedules.jl` match the Python/Rust suites, which keeps the
three implementations at parity to 1e-9.

All schedules are pure functions of the integer step `t` and a few
hyperparameters. Steps are zero-indexed: `t = 0` is the first update and
`t = total_steps` is the final point of the horizon.
"""
module Ch198LrSchedules

export cosine_annealing, linear_warmup, warmup_cosine, one_cycle,
    one_cycle_momentum, schedule_curve

function _check_horizon(total_steps::Integer, name::AbstractString)
    total_steps >= 1 || throw(ArgumentError("$name must be >= 1, got $total_steps"))
    return nothing
end

function _check_step(t::Integer)
    t >= 0 || throw(ArgumentError("t must be >= 0, got $t"))
    return nothing
end

function _check_bounds(eta_max::Real, eta_min::Real)
    eta_max >= eta_min ||
        throw(ArgumentError("eta_max ($eta_max) must be >= eta_min ($eta_min)"))
    eta_min >= 0 || throw(ArgumentError("eta_min must be >= 0, got $eta_min"))
    return nothing
end

"""
    cosine_annealing(t, total_steps, eta_max=0.1, eta_min=0.0)

Cosine annealing learning rate at step `t` over a horizon of `total_steps`: half a
cosine wave from `eta_max` at `t = 0` down to `eta_min` at `t = total_steps`. Steps
beyond the horizon clamp to `eta_min`.
"""
function cosine_annealing(t::Integer, total_steps::Integer,
        eta_max::Real=0.1, eta_min::Real=0.0)
    _check_step(t)
    _check_horizon(total_steps, "total_steps")
    _check_bounds(eta_max, eta_min)
    t >= total_steps && return float(eta_min)
    c = cos(pi * t / total_steps)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + c)
end

"""
    linear_warmup(t, warmup_steps, eta_target, eta_start=0.0)

Linear warmup from `eta_start` to `eta_target` over `warmup_steps`. At `t = 0`
returns `eta_start`; at `t >= warmup_steps` returns `eta_target`.
"""
function linear_warmup(t::Integer, warmup_steps::Integer,
        eta_target::Real, eta_start::Real=0.0)
    _check_step(t)
    _check_horizon(warmup_steps, "warmup_steps")
    (eta_target >= 0 && eta_start >= 0) ||
        throw(ArgumentError("learning rates must be non-negative"))
    t >= warmup_steps && return float(eta_target)
    frac = t / warmup_steps
    return eta_start + (eta_target - eta_start) * frac
end

"""
    warmup_cosine(t, warmup_steps, total_steps, eta_max, eta_min=0.0, eta_start=0.0)

Linear warmup followed by cosine annealing: the standard modern recipe. Ramps
linearly from `eta_start` to `eta_max` over the warmup, then cosine-descends to
`eta_min`. Requires `0 <= warmup_steps < total_steps`.
"""
function warmup_cosine(t::Integer, warmup_steps::Integer, total_steps::Integer,
        eta_max::Real, eta_min::Real=0.0, eta_start::Real=0.0)
    _check_step(t)
    _check_horizon(total_steps, "total_steps")
    _check_bounds(eta_max, eta_min)
    eta_start >= 0 || throw(ArgumentError("eta_start must be >= 0, got $eta_start"))
    (0 <= warmup_steps < total_steps) || throw(ArgumentError(
        "warmup_steps must satisfy 0 <= warmup_steps < total_steps=$total_steps, got $warmup_steps"))
    if t < warmup_steps
        frac = t / warmup_steps
        return eta_start + (eta_max - eta_start) * frac
    end
    t >= total_steps && return float(eta_min)
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    c = cos(pi * progress)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + c)
end

"""
    one_cycle(t, total_steps, eta_max, eta_min=nothing, pct_start=0.3)

One-cycle learning rate (Smith and Topin) with cosine ramps. Rises from `eta_min`
to `eta_max` over the first `pct_start` fraction, then descends back, both legs
half-cosines. `eta_min=nothing` defaults to `eta_max / 25`. `pct_start` in `(0, 1)`.
"""
function one_cycle(t::Integer, total_steps::Integer, eta_max::Real,
        eta_min=nothing, pct_start::Real=0.3)
    _check_step(t)
    _check_horizon(total_steps, "total_steps")
    emin = eta_min === nothing ? eta_max / 25.0 : float(eta_min)
    _check_bounds(eta_max, emin)
    (0 < pct_start < 1) || throw(ArgumentError("pct_start must be in (0, 1), got $pct_start"))
    t1 = pct_start * total_steps
    t >= total_steps && return float(emin)
    if t <= t1
        c = cos(pi * t / t1)
        return emin + 0.5 * (eta_max - emin) * (1.0 - c)
    end
    progress = (t - t1) / (total_steps - t1)
    c = cos(pi * progress)
    return emin + 0.5 * (eta_max - emin) * (1.0 + c)
end

"""
    one_cycle_momentum(t, total_steps, mom_max=0.95, mom_min=0.85, pct_start=0.3)

Antiphase momentum schedule for the one-cycle policy: falls from `mom_max` to
`mom_min` while the learning rate climbs, then rises back. Both momenta in `[0, 1)`
with `mom_max >= mom_min`.
"""
function one_cycle_momentum(t::Integer, total_steps::Integer,
        mom_max::Real=0.95, mom_min::Real=0.85, pct_start::Real=0.3)
    _check_step(t)
    _check_horizon(total_steps, "total_steps")
    (0 <= mom_min <= mom_max < 1) || throw(ArgumentError(
        "momenta must satisfy 0 <= mom_min <= mom_max < 1, got mom_min=$mom_min, mom_max=$mom_max"))
    (0 < pct_start < 1) || throw(ArgumentError("pct_start must be in (0, 1), got $pct_start"))
    t1 = pct_start * total_steps
    t >= total_steps && return float(mom_max)
    if t <= t1
        c = cos(pi * t / t1)
        return mom_max - 0.5 * (mom_max - mom_min) * (1.0 - c)
    end
    progress = (t - t1) / (total_steps - t1)
    c = cos(pi * progress)
    return mom_max - 0.5 * (mom_max - mom_min) * (1.0 + c)
end

"""
    schedule_curve(name, total_steps; warmup_steps=0, eta_max=0.1, eta_min=0.0, pct_start=0.3)

Materialize a schedule over all steps `0, 1, ..., total_steps - 1`. `name` is one
of `"cosine"`, `"warmup_cosine"`, or `"one_cycle"`.
"""
function schedule_curve(name::AbstractString, total_steps::Integer;
        warmup_steps::Integer=0, eta_max::Real=0.1, eta_min::Real=0.0,
        pct_start::Real=0.3)
    _check_horizon(total_steps, "total_steps")
    if name == "cosine"
        return [cosine_annealing(t, total_steps, eta_max, eta_min) for t in 0:(total_steps - 1)]
    elseif name == "warmup_cosine"
        return [warmup_cosine(t, warmup_steps, total_steps, eta_max, eta_min, 0.0)
                for t in 0:(total_steps - 1)]
    elseif name == "one_cycle"
        return [one_cycle(t, total_steps, eta_max, eta_min, pct_start)
                for t in 0:(total_steps - 1)]
    else
        throw(ArgumentError(
            "unknown schedule name \"$name\"; expected 'cosine', 'warmup_cosine', or 'one_cycle'"))
    end
end

end # module Ch198LrSchedules
