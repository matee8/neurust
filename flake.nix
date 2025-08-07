{
    description = "Batteries included crate for building, training and running neural networks.";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
        crane.url = "github:ipetkov/crane";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = inputs:
        inputs.flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = inputs.nixpkgs.legacyPackages.${system};

                craneLib = inputs.crane.mkLib pkgs;

                commonArgs = {
                    src = craneLib.cleanCargoSource ./.;
                    strictDeps = true;
                };

                neurust = craneLib.buildPackage (
                    commonArgs
                    // {
                        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
                    }
                );
            in
            {
                checks = {
                    inherit neurust;
                };

                packages.default = neurust;

                apps.default = inputs.flake-utils.lib.mkApp {
                    drv = neurust;
                };

                devShells.default = craneLib.devShell {
                    checks = inputs.self.checks.${system};

                    packages = [
                        pkgs.rust-analyzer
                    ];
                };

            }
        );
}
