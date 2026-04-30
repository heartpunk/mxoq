{
  description = "Binary-semantics extraction targets (libtsm v4.4.2 + libmd) — matches the recipe used for the README walkthroughs' empirical validation.";

  # Pinned via flake.lock; a clean clone of this gist + `nix build` reproduces
  # bit-identically. Floats here only at lock-update time.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      # Both targets are Linux-shared-object extraction targets. On macOS /
      # other host platforms, build via a Linux remote builder
      # (`nix build --system x86_64-linux`) or inside a Linux VM.
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Override nixpkgs.libtsm to:
      # (1) pin the version that matches our empirical validation (4.4.2);
      # (2) apply the extraction-target compile flags.
      #
      # `-O0 -fno-inline` preserves all static functions as separate symbols
      # so the per-function CFG extraction sees them as discrete entry points;
      # `-Wno-error` silences warnings the upstream build promotes to errors
      # at this version-flag combination.
      libtsm = pkgs.libtsm.overrideAttrs (_old: {
        version = "4.4.2";
        src = pkgs.fetchFromGitHub {
          owner = "kmscon";
          repo = "libtsm";
          tag = "v4.4.2";
          hash = "sha256-DWy7kgBbXUEt2Htcugo8PaVoHE23Nu22EIrB5f6/P30=";
        };
        NIX_CFLAGS_COMPILE = "-O0 -fno-inline -Wno-error";
      });

      # libmd: BSD message-digest functions (the second walkthrough's target
      # exercises SHA1Update from this library). No override needed —
      # nixpkgs's stock build at the flake.lock-pinned version is what the
      # gist's regen pipeline extracts the LTS from, so user reproduction
      # via `nix build .#libmd` lands on a byte-identical .so. We don't
      # vendor libmd source or binary; the BSD-3 license travels with
      # the upstream nixpkgs derivation, so no copyright file is shipped
      # in this repo.
      libmd = pkgs.libmd;
    in {
      packages.${system} = {
        default = libtsm;
        libtsm = libtsm;
        libmd = libmd;
      };
    };
}
