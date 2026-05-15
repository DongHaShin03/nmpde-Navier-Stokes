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
    // ----- QUI AGGIUNGERE METODI NON LINEARI -----
    // Se nasce un nuovo metodo, aggiungerlo qui, nel parser sotto e nel .prm.
    // Dopo il cambio, NavierStokes::run() deve avere un ramo dedicato nel loop.
    None,
    Picard,
    PicardRelaxed,
    Newton,
    NewtonDamped
};

inline NonlinearMethod parse_nonlinear_method(const std::string &value)
{
    const std::string key = normalize_option(value);
    if (key == "none")
        return NonlinearMethod::None;
    if (key == "picard")
        return NonlinearMethod::Picard;
    if (key == "picard_relaxed")
        return NonlinearMethod::PicardRelaxed;
    if (key == "newton")
        return NonlinearMethod::Newton;
    if (key == "newton_damped")
        return NonlinearMethod::NewtonDamped;

    throw std::invalid_argument("Unknown nonlinear method: " + value);
}

inline const char *to_string(const NonlinearMethod method)
{
    switch (method)
    {
        case NonlinearMethod::None:
            return "none";
        case NonlinearMethod::Picard:
            return "picard";
        case NonlinearMethod::PicardRelaxed:
            return "picard_relaxed";
        case NonlinearMethod::Newton:
            return "newton";
        case NonlinearMethod::NewtonDamped:
            return "newton_damped";
    }

    return "unknown";
}

enum class PreconditionerKind
{
    // ----- QUI AGGIUNGERE PRECONDIZIONATORI -----
    // Se nasce un nuovo precondizionatore, aggiungerlo qui, nel parser sotto,
    // in PreconditionerFactory.hpp e nella selezione ammessa dal .prm.
    Identity,
    Simple,
    BlockDiagonal,
    BlockTriangular,
    Yosida,
    PCD
};

inline PreconditionerKind parse_preconditioner_kind(const std::string &value)
{
    const std::string key = normalize_option(value);
    if (key == "identity")
        return PreconditionerKind::Identity;
    if (key == "simple")
        return PreconditionerKind::Simple;
    if (key == "block_diagonal")
        return PreconditionerKind::BlockDiagonal;
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
        case PreconditionerKind::Identity:
            return "identity";
        case PreconditionerKind::Simple:
            return "simple";
        case PreconditionerKind::BlockDiagonal:
            return "block_diagonal";
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
    bool pspg = false;
};

#endif
