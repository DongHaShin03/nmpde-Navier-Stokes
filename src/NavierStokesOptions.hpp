#ifndef NAVIER_STOKES_OPTIONS_HPP
#define NAVIER_STOKES_OPTIONS_HPP

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

inline std::string normalize_option(std::string value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](const unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });

    std::replace(value.begin(), value.end(), '-', '_');
    std::replace(value.begin(), value.end(), ' ', '_');
    return value;
}

enum class NonlinearMethod
{
    // One Oseen linearization per time step: beta = u^n.
    Oseen,
    // Picard fixed-point iterations per time step: beta = u^{k}.
    Picard,
    // Relaxed Picard: u^{k+1} <- (1-alpha)u^k + alpha*u_raw.
    PicardRelaxed
};

inline NonlinearMethod parse_nonlinear_method(const std::string &value)
{
    const std::string key = normalize_option(value);
    if (key == "none" || key == "oseen")
        return NonlinearMethod::Oseen;
    if (key == "picard")
        return NonlinearMethod::Picard;
    if (key == "picard_relaxed")
        return NonlinearMethod::PicardRelaxed;

    throw std::invalid_argument("Unknown nonlinear method: " + value);
}

inline const char *to_string(const NonlinearMethod method)
{
    switch (method)
    {
        case NonlinearMethod::Oseen:
            return "oseen";
        case NonlinearMethod::Picard:
            return "picard";
        case NonlinearMethod::PicardRelaxed:
            return "picard_relaxed";
    }

    return "unknown";
}

enum class PreconditionerKind
{
    Simple,
    BlockTriangular,
    Yosida,
    PCD
};

inline PreconditionerKind parse_preconditioner_kind(const std::string &value)
{
    const std::string key = normalize_option(value);
    if (key == "simple")
        return PreconditionerKind::Simple;
    if (key == "block_triangular")
        return PreconditionerKind::BlockTriangular;
    if (key == "yosida")
        return PreconditionerKind::Yosida;
    if (key == "pcd")
        return PreconditionerKind::PCD;

    throw std::invalid_argument("Unknown preconditioner: " + value);
}

inline const char *to_string(const PreconditionerKind preconditioner)
{
    switch (preconditioner)
    {
        case PreconditionerKind::Simple:
            return "simple";
        case PreconditionerKind::BlockTriangular:
            return "block_triangular";
        case PreconditionerKind::Yosida:
            return "yosida";
        case PreconditionerKind::PCD:
            return "pcd";
    }

    return "unknown";
}

struct StabilizationOptions
{
    bool temam = true;
    bool grad_div = false;
    double gamma_grad_div = 0.0;
    bool supg = false;
};

struct PreconditionerIterationOptions
{
    unsigned int block_triangular_velocity_max_iterations = 100;
    unsigned int block_triangular_schur_max_iterations = 250;
    double block_triangular_velocity_relative_tolerance = 1e-2;
    double block_triangular_schur_relative_tolerance = 1e-3;
    unsigned int simple_velocity_max_iterations = 5;
    unsigned int simple_schur_max_iterations = 20;
    double simple_velocity_relative_tolerance = 1e-2;
    double simple_schur_relative_tolerance = 1e-3;
    unsigned int pcd_velocity_max_iterations = 10;
    unsigned int pcd_pressure_max_iterations = 20;
    double pcd_velocity_relative_tolerance = 1e-2;
    double pcd_pressure_relative_tolerance = 1e-3;
    unsigned int yosida_velocity_max_iterations = 100000;
    unsigned int yosida_schur_max_iterations = 100000;
    unsigned int yosida_correction_max_iterations = 100000;
    double yosida_relative_tolerance = 1e-2;
    double preconditioner_absolute_tolerance = 1e-12;
    double yosida_absolute_tolerance = 1e-14;
};

#endif
