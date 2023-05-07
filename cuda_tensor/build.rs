use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
extern crate cc;

fn main() {
    let nvcc      = "/usr/local/cuda/bin/nvcc";
    let cu_files:Vec<&str> = vec![ "kernels/kernel.cu" ];
    let mut ptx_files:Vec<String> = vec!();
    let sm_ver = "61";

    println!("OUT_DIR {}", env::var("OUT_DIR").unwrap());
    let out_dir = env::var("OUT_DIR").unwrap();
    for cu_file in cu_files.iter() {
	let file_pathbuf = PathBuf::from(cu_file);
	let output_path = out_dir.clone() + "/" + file_pathbuf.file_stem().unwrap().to_str().unwrap() + ".fatbin";
	let output = Command::new(nvcc)
	    .arg("-fatbin")
	    .arg("-gencode")
	    .arg(format!("arch=compute_{sm_ver},code=sm_{sm_ver}"))
            .arg(cu_file)
	    .arg("-o")
	    .arg(&output_path)
	    .output()
	    .expect("failed to spawn process");
	if !output.status.success() {
	    std::io::stderr().write_all(&output.stderr).unwrap();
	    assert!(output.status.success());
	}
	ptx_files.push(output_path);
    }
    println!("compile cu files to fatbin\n");

    println!("cargo:rustc-link-arg=-L/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudnn");
    println!("cargo:rustc-link-lib=cudart");

}
