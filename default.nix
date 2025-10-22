{
  pkgs ?
    import <nixpkgs> {
      overlays = [
        (final: prev: {
          opencv4 = prev.opencv4.override {
            enableGtk3 = true;
            enablePython = true;
          };
        })
      ];
    },
}: let
  pythonWithPkgs = pkgs.python313.withPackages (ppkgs: [
    ppkgs.numpy
    ppkgs.pymupdf
    ppkgs.rarfile
    ppkgs.matplotlib
    ppkgs.pyyaml
    ppkgs.ultralytics
    ppkgs.opencv4
    ppkgs.pillow
    ppkgs.jupyter
    ppkgs.ipykernel
    ppkgs.jupyter-cache
  ]);

  myPackages = with pkgs; [
    jupyter
    pythonWithPkgs
    gfortran.cc.lib
  ];
in
  pkgs.mkShell {
    buildInputs = myPackages;

    shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.gfortran.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

            # Add our custom Python to PATH so 'python' command uses it
            export PATH="${pythonWithPkgs}/bin:$PATH"

            # Create a wrapper script that will always resolve to the correct Python path
            KERNEL_WRAPPER_DIR="$HOME/.local/share/jupyter/kernels_wrappers"
            mkdir -p "$KERNEL_WRAPPER_DIR"

            # Create a wrapper script for the Python kernel
            cat > "$KERNEL_WRAPPER_DIR/python_kernel_wrapper.sh" << 'EOF'
      #!/bin/sh
      # This wrapper finds the python from PATH at runtime
      exec python -m ipykernel_launcher -f "$1"
      EOF
            chmod +x "$KERNEL_WRAPPER_DIR/python_kernel_wrapper.sh"

            # Manual kernel setup with the wrapper script
            PYTHON_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/python-nix"
            mkdir -p "$PYTHON_KERNEL_DIR"

            cat > "$PYTHON_KERNEL_DIR/kernel.json" << EOF
      {
        "argv": [
          "$KERNEL_WRAPPER_DIR/python_kernel_wrapper.sh",
          "{connection_file}"
        ],
        "display_name": "Python 3 (Nix Environment)",
        "language": "python",
        "env": {
          "PATH": "${pythonWithPkgs}/bin:$PATH"
        }
      }
      EOF

            echo "Python kernel configured with wrapper script"
            echo "Environment ready with GCC fortran fix and Python kernel"
            echo "Python with packages available at: ${pythonWithPkgs}/bin/python"

            # List kernels to verify
            jupyter kernelspec list || echo "Jupyter command might not be available yet"
    '';
  }
