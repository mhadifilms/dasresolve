// Copyright 2026 The DasGrain OFX Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Currently a placeholder TU so CMake's add_library has a non-header source
// for this submodule and so we can link param-related helpers later without
// forcing a header-only dependency.

#include "Params.h"

namespace dasgrain {
// Reserved for parameter-construction helpers that we don't want to inline
// (e.g. group / page setup) once the describeInContext gets large.
}  // namespace dasgrain
