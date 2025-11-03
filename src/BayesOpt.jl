"""
Example

```
using BayesOpt
config = ConfigParameters()         # calls initialize_parameters_to_default of the C API
set_kernel!(config, "kMaternARD5")  # calls set_kernel of the C API
config.sc_type = SC_MAP
f(x) = sum(x .^ 2)
lowerbound = [-2., -2.]; upperbound = [2., 2.]
optimizer, optimum = bayes_optimization(f, lowerbound, upperbound, config)
```

Exports: `KernelParameters`, `MeanParameters`, `ConfigParameters`, `bayes_optimization`,
`bayes_optimization_disc`, `bayes_optimization_categorical`, `set_kernel!`, `set_mean!`,
`set_criteria!`, `set_surrogate!`, `set_log_file!`, `set_load_file!`, `set_save_file!`,
`set_learning!`, `set_score!`, `L_FIXED`, `L_EMPIRICAL`, `L_DISCRETE`, `L_MCMC`, `L_ERROR`,
`SC_MTL`, `SC_ML`, `SC_MAP`, `SC_LOOCV`, `SC_ERROR`


 

See also https://rmcantin.bitbucket.io/html/usemanual.html
"""
module BayesOpt

using Artifacts
using Libdl
using Base.BinaryPlatforms

# library path
global libbayesopt = ""
global libnlopt = ""

function __init__()

    platform = Base.BinaryPlatforms.HostPlatform()

    tags = platform.tags

    artifact_name = if get(tags, "os", "") == "linux" && get(tags, "arch", "") == "aarch64" && get(tags, "libc", "") == "glibc"
        "BayesOptBuilder_aarch64_linux_gnu"
    elseif get(tags, "os", "") == "linux" && get(tags, "arch", "") == "armv7l" && get(tags, "libc", "") == "glibc" && get(tags, "call_abi", "") == "eabihf"
        "BayesOptBuilder_arm_linux_gnueabihf"
    elseif get(tags, "os", "") == "linux" && get(tags, "arch", "") == "i686" && get(tags, "libc", "") == "glibc"
        "BayesOptBuilder_i686_linux_gnu"
    elseif get(tags, "os", "") == "linux" && get(tags, "arch", "") == "powerpc64le" && get(tags, "libc", "") == "glibc"
        "BayesOptBuilder_powerpc64le_linux_gnu"
    elseif get(tags, "os", "") == "macos" && get(tags, "arch", "") == "x86_64"
        "BayesOptBuilder_x86_64_apple_darwin14"
    elseif get(tags, "os", "") == "linux" && get(tags, "arch", "") == "x86_64" && get(tags, "libc", "") == "glibc"
        "BayesOptBuilder_x86_64_linux_gnu"
    else
        error("Your platform ($platform) is not supported by this package! Tags: $tags")
    end

    toml_path = joinpath(@__DIR__, "..", "Artifacts.toml")

    if !isfile(toml_path)
         toml_path = joinpath(@__DIR__, "Artifacts.toml")
         if !isfile(toml_path)
             error("Could not find Artifacts.toml. Looked in $(joinpath(@__DIR__, "..")) and $(@__DIR__)")
         end
    end
    artifact_hash_val = Artifacts.artifact_hash(artifact_name, toml_path)
    
    if isnothing(artifact_hash_val)
        error("Could not find artifact entry for '$artifact_name' in $toml_path")
    end

    artifact_dir = Artifacts.artifact_path(artifact_hash_val)

    lib_dir = joinpath(artifact_dir, "lib")

    global libbayesopt = Libdl.find_library(["libbayesopt"], [lib_dir])
    global libnlopt = Libdl.find_library(["libnlopt"], [lib_dir])

    if isempty(libbayesopt)
        error("Could not find 'libbayesopt' in artifact directory. Please re-build package.")
    end
end

export KernelParameters, MeanParameters, ConfigParameters, bayes_optimization,
bayes_optimization_disc, bayes_optimization_categorical

import Base: show
struct KernelParameters
    name::Cstring
    hp_mean::NTuple{128, Cdouble}
    hp_std::NTuple{128, Cdouble}
    n_hp::Csize_t
end
function show(io::IO, mime::MIME"text/plain", o::KernelParameters)
    println(io, "$(unsafe_string(o.name)) Kernel")
    if o.n_hp > 0
        println(io, "  hyperparameters mean $(o.hp_mean[1:o.n_hp])")
        println(io, "  hyperparameters std $(o.hp_std[1:o.n_hp])")
    end
end

@enum LearningType::Cint begin
    L_FIXED
    L_EMPIRICAL
    L_DISCRETE
    L_MCMC
    L_ERROR = -1
end
export L_FIXED, L_EMPIRICAL, L_DISCRETE, L_MCMC, L_ERROR

@enum ScoreType::Cint begin
    SC_MTL
    SC_ML
    SC_MAP
    SC_LOOCV
    SC_ERROR = -1
end
export SC_MTL, SC_ML, SC_MAP, SC_LOOCV, SC_ERROR

struct MeanParameters
    name::Cstring
    coef_mean::NTuple{128, Cdouble}
    coef_std::NTuple{128, Cdouble}
    n_coef::Csize_t
end
function show(io::IO, mime::MIME"text/plain", o::MeanParameters)
    println(io, "$(unsafe_string(o.name)) Mean")
    if o.n_coef > 0
        println(io, "  coefficients mean $(o.coef_mean[1:o.n_coef])")
        println(io, "  coefficients std $(o.coef_std[1:o.n_coef])")
    end
end

mutable struct ConfigParameters
    n_iterations::Csize_t
    n_inner_iterations::Csize_t
    n_init_samples::Csize_t
    n_iter_relearn::Csize_t
    init_method::Csize_t          
    random_seed::Cint
    verbose_level::Cint
    log_filename::Cstring
    load_save_flag::Csize_t
    load_filename::Cstring
    save_filename::Cstring
    surr_name::Cstring
    sigma_s::Cdouble
    noise::Cdouble
    alpha::Cdouble
    beta::Cdouble
    sc_type::ScoreType
    l_type::LearningType
    l_all::Cint
    epsilon::Cdouble
    force_jump::Csize_t
    kernel::KernelParameters
    mean::MeanParameters
    crit_name::Cstring
    crit_params::NTuple{128, Cdouble}
    n_crit_params::Csize_t
end
"""
    ConfigParameters()

Returns default parameters of BayesOpt (see initialize_parameters_to_default in the C API).
"""
ConfigParameters() = ccall((:initialize_parameters_to_default, libbayesopt), ConfigParameters, ())

function show(io::IO, mime::MIME"text/plain", o::ConfigParameters)
    println(io, "ConfigParameters")
    for field in fieldnames(ConfigParameters)
        val = getfield(o, field) 
        if field == :crit_params
            o.n_crit_params == 0 && continue
            println(io, "$field = $(val[1:o.n_crit_params])")
        end
        valtoshow = typeof(val) == Cstring ? unsafe_string(val) : val
        if typeof(val) == KernelParameters || typeof(val) == MeanParameters
            show(io, mime, val)
        else
            println(io, "$field = $valtoshow")
        end
    end
end

for func in [:set_kernel, :set_mean, :set_criteria, :set_surrogate, :set_log_file, 
             :set_load_file, :set_save_file, :set_learning, :set_score]
    jfuncname = Symbol(func, "!")
    jfuncstring = split(string(jfuncname), ".")[end]
    @eval begin
        $jfuncname(config, name) = ccall(($(string(func)), libbayesopt), Cvoid, (Ptr{ConfigParameters}, Cstring), Ref(config), name)
        @doc "`$($jfuncstring)(config, name)` $(replace($(string(func)), "_" => " ")) in `config` to `name`." $jfuncname
        export $jfuncname
    end
end

@inline function prepare_cargs(func, n)
    optimizer = zeros(n); optimum  = Ref{Cdouble}(0)
    bofunc = (n, x, g, d) -> func(unsafe_wrap(Array, x, n))
    cfunc = @cfunction $bofunc Cdouble (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}) 
    cfunc, optimizer, optimum
end

"""
    bayes_optimization(func, lb, ub, config)

Runs continuous Bayesian optimization on `func` that takes vectors of length `d`
as argument and returns a real number, within the box defined by the lowerbounds
`lb` (a vector of length `d` with lowerbounds for each dimension) and upperbounds
`ub`, using `config` (see `ConfigParameters`).
"""
function bayes_optimization(func, lb, ub, config)
    n = length(lb)
    length(ub) == n || @error("lowerbounds and upperbounds have different length.")
    cfunc, optimizer, optimum = prepare_cargs(func, n)
    ccall((:bayes_optimization, libbayesopt), Cint, 
          (Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, 
           Ptr{Cdouble}, Ptr{Cdouble}, ConfigParameters), 
          n, cfunc, Ptr{Nothing}(0), 
          pointer(lb), pointer(ub), pointer(optimizer), optimum, config)
    (optimizer = optimizer, optimum = optimum.x)
end

"""
    bayes_optimization_disc(func, valid_x, config)

Runs  Bayesian optimization on `func` that takes vectors of length `d`
as argument and returns a real number, on `valid_x` an array of valid points 
(vectors of length `d`) using `config` (see `ConfigParameters`).
"""
function bayes_optimization_disc(func, xs::Array{<:Array{<:Number, 1}, 1}, config)
    n_points = length(xs)
    n = length(xs[1])
    valid_x = pointer(vcat(xs...))
    cfunc, optimizer, optimum = prepare_cargs(func, n)
    ccall((:bayes_optimization_disc, libbayesopt), Cint, 
          (Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, 
           Ptr{Cdouble}, Ptr{Cdouble}, ConfigParameters), 
          n, cfunc, Ptr{Nothing}(0), 
          valid_x, Csize_t(n_points), pointer(optimizer), optimum, config)
    (optimizer = optimizer, optimum = optimum.x)
end

"""
    bayes_optimization_categorical(func, categories, config)

Runs Bayesian optimization on `func` that takes vectors of length `d` as
argument and returns a real number, with `categories` array of size `d` with the
number of categories per dimension, using `config` (see `ConfigParameters`).
"""
function bayes_optimization_categorical(func, categories::Array{Integer, 1}, config)
    n = length(categories)
    cfunc, optimizer, optimum = prepare_cargs(func, n)
    ccall((:bayes_optimization_categorical, libbayesopt), Cint, 
          (Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, 
           Ptr{Cdouble}, Ptr{Cdouble}, ConfigParameters), 
          n, cfunc, Ptr{Nothing}(0), 
          pointer(categories), pointer(optimizer), optimum, config)
    (optimizer = optimizer, optimum = optimum.x)
end
end
