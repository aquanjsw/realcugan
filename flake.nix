{
  description = "RealCUGAN with Python CLI";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";

    weights = {
      type = "tarball";
      url = "https://github.com/aquanjsw/realcugan/releases/download/weights/weights.tar.gz";
      flake = false;
    };
  };

  outputs =
    { self, nixpkgs, ... }@inputs:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
      packageOverrides = self: super: {
      };
      genPythonEnv =
        pkgs:
        let
          python = pkgs.python3.override {
            inherit packageOverrides;
            self = python;
          };
        in
        python.withPackages (
          ps:
          (with ps; [
            moviepy
            numpy
            opencv-python-headless
            torch
            loguru
            tqdm
          ])
        );
      genTargetPkgs =
        pkgs: pythonEnv:
        (
          pkgs:
          (
            [
              pythonEnv
            ]
            ++ (with pkgs; [
              cudatoolkit
              stdenv.cc
              libxcb
              libGL
              glib
            ])
          )
        );

      getPkgs =
        system:
        (import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        });
    in
    {
      devShells = forAllSystems (system: {
        default =
          let
            pkgs = getPkgs system;
            pythonEnv = genPythonEnv pkgs;
          in
          (pkgs.buildFHSEnv {
            name = "realcugan-dev";
            targetPkgs = genTargetPkgs pkgs pythonEnv;
            runScript = pkgs.writeShellScript "run" ''
              mkdir -p .zed
              ln -sf ${pythonEnv}/bin/python .zed/python
              ln -sfT ${inputs.weights} weights
              exec bash
            '';
          }).env;
      });

      packages = forAllSystems (system: {
        default =
          let
            pkgs = getPkgs system;
            inherit (pkgs) lib;
            pythonEnv = genPythonEnv pkgs;
          in
          pkgs.stdenv.mkDerivation {
            name = "realcugan";
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.intersection (lib.fileset.gitTracked ./.) (
                lib.fileset.unions [
                  ./main.py
                  ./args.py
                  ./models
                ]
              );
            };
            installPhase = ''
              mkdir -p $out/bin $out/lib
              cp -r *.py models $out/lib/
              ln -sfT ${inputs.weights} $out/lib/weights
              makeWrapper ${lib.getBin pythonEnv}/bin/python $out/bin/realcugan \
                --add-flags "$out/lib/main.py"
            '';
            nativeBuildInputs = [ pkgs.makeWrapper ];
            buildInputs = [ pythonEnv ];
          };
      });

      overlays.default = final: prev: {
        realcugan = self.packages.${final.stdenv.hostPlatform.system}.default;
      };
    };
}
