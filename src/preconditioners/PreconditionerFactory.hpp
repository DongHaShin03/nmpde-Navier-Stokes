#ifndef PRECONDITIONER_FACTORY_HPP
#define PRECONDITIONER_FACTORY_HPP

#include "../NavierStokesOptions.hpp"

#include "BlockDiagonal.hpp"
#include "BlockTriangular.hpp"
#include "Identity.hpp"
#include "PCD.hpp"
#include "Simple.hpp"
#include "Yosida.hpp"

#include <memory>
#include <stdexcept>

inline std::unique_ptr<NavierStokesPreconditioner>
make_preconditioner(const PreconditionerKind preconditioner)
{
    switch (preconditioner)
    {
        case PreconditionerKind::Identity:
            return std::make_unique<IdentityPreconditioner>();
        case PreconditionerKind::Simple:
            return std::make_unique<Simple>();
        case PreconditionerKind::BlockDiagonal:
            return std::make_unique<BlockDiagonal>();
        case PreconditionerKind::BlockTriangular:
            return std::make_unique<BlockTriangular>();
        case PreconditionerKind::Yosida:
            return std::make_unique<Yosida>();
        case PreconditionerKind::PCD:
            return std::make_unique<PCD>();
    }

    throw std::runtime_error("Unknown preconditioner kind.");
}

#endif
