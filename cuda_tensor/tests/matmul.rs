use cuda_binding::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;
use std::rc::Rc;
use std::time::{Duration, Instant};

fn matmul(x: &[f32], y: &[f32], (m, l, n): (usize, usize, usize)) -> Vec<f32> {
    let mut z: Vec<f32> = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..l {
                z[i * n + j] += x[i * l + k] * y[k * n + j]; //x[i][k]*y[k][j]
            }
        }
    }
    z
}

#[test]
fn tensor_matmul() -> Result<(), CUDAError> {
    let mut rng = Pcg64::from_entropy();
    let uniform_dist: Uniform<f32> = Uniform::new(-2.0, 2.0);

    let result = cuda::initailize();
    let count_of_device = cuda::Device::get_count()?;
    if count_of_device < 1 {
        println!("unable to initialize cuda. {}", result);
        return Err(CUDAError::NoDevice);
    }
    println!("find num of cuda devices: {}\n", count_of_device);

    let cuda_device = cuda::Device::new(0)?;

    let cap = cuda_device.get_compute_capability()?;
    println!("Compute Capalibiliry. {}.{}", cap.0, cap.1);
    println!("Device name: {}.", cuda_device.get_name()?);

    let cuda_context = cuda::Context::new(cuda_device, cuda::context::flags::MapHost)?;
    if false {
        const M: usize = 3;
        const L: usize = 4;
        const N: usize = 2;

        let tensor_x_src: Vec<f32> =
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 2.0];
        let tensor_y_src: Vec<f32> = vec![2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 5.0, 2.0];
        for i in 0..M {
            print!("[");
            for j in 0..L {
                print!("{} ", tensor_x_src[i * L + j]);
            }
            print!("]\n");
        }
        println!("");
        for i in 0..L {
            print!("[");
            for j in 0..N {
                print!("{} ", tensor_y_src[i * N + j]);
            }
            print!("]\n");
        }
        println!("");

        let start = Instant::now();
        let tensor_z_dst = matmul(&tensor_x_src, &tensor_y_src, (M, L, N));
        let end = start.elapsed();
        println!("cpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());

        for i in 0..M {
            print!("[");
            for j in 0..N {
                print!("{} ", tensor_z_dst[i * N + j]);
            }
            print!("]\n");
        }

        {
            let context = cuda_tensor::Context::new(Rc::clone(&cuda_context))?;
            let mut tensor_x = context.create_tensor::<f32>(&[M, L])?;
            tensor_x.set_elements(tensor_x_src)?;
            let mut tensor_y = context.create_tensor::<f32>(&[L, N])?;
            tensor_y.set_elements(tensor_y_src)?;
            let start = Instant::now();
            let tensor_z = cuda_tensor::tensor::matmul(&tensor_x, &tensor_y);
            let tensor_z_host = tensor_z.get_elements()?;
            let end = start.elapsed();
            println!("{}", tensor_z);
            println!("gpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());
            for (&z1, &z2) in tensor_z_host.iter().zip(tensor_z_dst.iter()) {
                if !((z1 - z2).abs() < 1.0e-4) {
                    println!("{}-{}", z1, z2);
                    assert!((z1 - z2).abs() < 1.0e-5);
                }
            }
        }
    }

    if false {
        const M: usize = 12;
        const L: usize = 9;
        const N: usize = 10;

	let tensor_x_src: Vec<f32> = (0..M * L)
            .map(|_| uniform_dist.sample(&mut rng))
            .collect::<Vec<f32>>();
        let tensor_y_src: Vec<f32> = (0..L * N)
            .map(|_| uniform_dist.sample(&mut rng))
            .collect::<Vec<f32>>();

        for i in 0..M {
            print!("[");
            for j in 0..L {
                print!("{} ", tensor_x_src[i * L + j]);
            }
            print!("]\n");
        }

        let start = Instant::now();
        let tensor_z_dst = matmul(&tensor_x_src, &tensor_y_src, (M, L, N));
        let end = start.elapsed();

        println!("cpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());

        for i in 0..M {
            print!("[");
            for j in 0..N {
                print!("{} ",tensor_z_dst[i*N+j]);
            }
            print!("]\n");
        }

	{
            let context = cuda_tensor::Context::new(Rc::clone(&cuda_context))?;
            let mut tensor_x = context.create_tensor::<f32>(&[M, L])?;
            tensor_x.set_elements(tensor_x_src)?;
            let mut tensor_y = context.create_tensor::<f32>(&[L, N])?;
            tensor_y.set_elements(tensor_y_src)?;
            let start = Instant::now();
            let tensor_z = cuda_tensor::tensor::matmul(&tensor_x, &tensor_y);
            let tensor_z_host = tensor_z.get_elements()?;
            let end = start.elapsed();
            println!("{}", tensor_z);
            println!("gpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());
            for (&z1, &z2) in tensor_z_host.iter().zip(tensor_z_dst.iter()) {
                if !((z1 - z2).abs() < 1.0e-4) {
                    println!("{}-{}", z1, z2);
                    assert!((z1 - z2).abs() < 1.0e-5);
                }
            }
        }
    }

    if true {
        const M: usize = 19;
        const L: usize = 20;
        const N: usize = 17;

	let tensor_x_src: Vec<f32> = (0..M * L)
            .map(|_| uniform_dist.sample(&mut rng))
            .collect::<Vec<f32>>();
        let tensor_y_src: Vec<f32> = (0..L * N)
            .map(|_| uniform_dist.sample(&mut rng))
            .collect::<Vec<f32>>();

        for i in 0..M {
            print!("[");
            for j in 0..L {
                print!("{} ", tensor_x_src[i * L + j]);
            }
            print!("]\n");
        }

        let start = Instant::now();
        let tensor_z_dst = matmul(&tensor_x_src, &tensor_y_src, (M, L, N));
        let end = start.elapsed();

        println!("cpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());

        for i in 0..M {
            print!("[");
            for j in 0..N {
                print!("{} ",tensor_z_dst[i*N+j]);
            }
            print!("]\n");
        }

	{
            let context = cuda_tensor::Context::new(Rc::clone(&cuda_context))?;
            let mut tensor_x = context.create_tensor::<f32>(&[M, L])?;
            tensor_x.set_elements(tensor_x_src)?;
            let mut tensor_y = context.create_tensor::<f32>(&[L, N])?;
            tensor_y.set_elements(tensor_y_src)?;
            let start = Instant::now();
            let tensor_z = cuda_tensor::tensor::matmul(&tensor_x, &tensor_y);
            let tensor_z_host = tensor_z.get_elements()?;
            let end = start.elapsed();
            println!("{}", tensor_z);
            println!("gpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());
            for (&z1, &z2) in tensor_z_host.iter().zip(tensor_z_dst.iter()) {
                if !((z1 - z2).abs() < 1.0e-4) {
                    println!("{}-{}", z1, z2);
                    assert!((z1 - z2).abs() < 1.0e-5);
                }
            }
        }
    }

    if true {
        const M: usize = 37;
        const L: usize = 128;
        const N: usize = 18;

        let tensor_x_src: Vec<f32> = (0..M * L)
            .map(|_| uniform_dist.sample(&mut rng))
            .collect::<Vec<f32>>();
        let tensor_y_src: Vec<f32> = (0..L * N)
            .map(|_| uniform_dist.sample(&mut rng))
            .collect::<Vec<f32>>();
/*
        for i in 0..M {
            print!("[");
            for j in 0..L {
                print!("{} ", tensor_x_src[i * L + j]);
            }
            print!("]\n");
        }
*/
        let start = Instant::now();
        let tensor_z_dst = matmul(&tensor_x_src, &tensor_y_src, (M, L, N));
        let end = start.elapsed();

        println!("cpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());
	/*
        for i in 0..M {
            print!("[");
            for j in 0..N {
                print!("{} ",tensor_z_dst[i*N+j]);
            }
            print!("]\n");
        }
*/
        {
            let context = cuda_tensor::Context::new(Rc::clone(&cuda_context))?;
            let mut tensor_x = context.create_tensor::<f32>(&[M, L])?;
            tensor_x.set_elements(tensor_x_src)?;
            let mut tensor_y = context.create_tensor::<f32>(&[L, N])?;
            tensor_y.set_elements(tensor_y_src)?;
            let start = Instant::now();
            let tensor_z = cuda_tensor::tensor::matmul(&tensor_x, &tensor_y);
            let tensor_z_host = tensor_z.get_elements()?;
            let end = start.elapsed();
	    println!("gpu matmul {}.{:09}.", end.as_secs(), end.subsec_nanos());
	    println!("{tensor_z}");

            for (i,(&z1, &z2)) in tensor_z_host.iter().zip(tensor_z_dst.iter()).enumerate() {
                if !((z1 - z2).abs() < 1.0e-4) {
                    println!("({} {}) {}-{}", i/N, i%N, z1, z2);
                    assert!((z1 - z2).abs() < 1.0e-5);
                }
            }
        }
    }

    Ok(())
}
