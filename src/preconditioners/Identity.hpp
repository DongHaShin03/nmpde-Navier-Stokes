#ifndef IDENTITY_PRECONDITIONER_HPP
#define IDENTITY_PRECONDITIONER_HPP

#include "NavierStokesPreconditioner.hpp"

class IdentityPreconditioner : public NavierStokesPreconditioner
{
    public:
        void initialize(const RequiredMatrices &) override {}

        void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            dst = src;
        }
};

#endif
