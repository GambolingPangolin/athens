{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    buildInputs = [ pkgs.python312 ];
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
}
