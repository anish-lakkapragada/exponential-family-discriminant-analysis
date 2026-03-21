import Lake
open Lake DSL

package «efda» where
  name := "efda"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «EFDAChallenge» where
  roots := #[`EFDAChallenge]
