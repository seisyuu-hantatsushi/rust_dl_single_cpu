
use cuda_binding::*;
//use cuda_tensor::*;
use std::rc::Rc;
use rand::SeedableRng;
use rand_distr::{Uniform,Distribution};
use rand_pcg::Pcg64;

#[test]
fn tensor_add() -> Result<(), CUDAError> {
    let result = cuda::initailize();

    if result != CUDAError::Success {
	eprintln!("unable to initialize cuda. {}", result);
	return Err(result);
    }

    let count_of_device = cuda::Device::getCount()?;
    if count_of_device < 1 {
	println!("unable to initialize cuda. {}", result);
	return Err(CUDAError::NoDevice);
    }
    println!("find num of cuda devices: {}\n", count_of_device);

    let cuda_device = cuda::Device::new(0)?;

    let cap = cuda_device.getComputeCapability()?;

    println!("Compute Capalibiliry. {}.{}", cap.0, cap.1);
    println!("Device name: {}.", cuda_device.get_name()?);
    let cuda_context = cuda::Context::new(cuda_device, cuda::context::flags::MapHost)?;

    {
	const M:usize = 7;
	const N:usize = 5;

	let context = cuda_tensor::Context::new(Rc::clone(&cuda_context))?;

	let tensor_x_src = (0..M*N).map(|c| { c as f32 }).collect::<Vec<f32>>();
	let mut tensor_x = context.create_tensor::<f32>(&[M,N])?;
	tensor_x.set_elements(tensor_x_src.clone())?;

	let tensor_y_src = (0..M*N).map(|c| { (c*2) as f32 }).collect::<Vec<f32>>();
	let mut tensor_y = context.create_tensor::<f32>(&[M,N])?;
	tensor_y.set_elements(tensor_y_src.clone())?;

	//println!("{}",tensor_x);
	//println!("{}",tensor_y);

	let tensor_z = cuda_tensor::tensor::add(&tensor_x, &tensor_y);

	//println!("{}",tensor_z);
	let tensor_z_host = tensor_z.get_elements()?;

	for ((&x,&y),&z) in tensor_x_src.iter().zip(tensor_y_src.iter()).zip(tensor_z_host.iter())
	{
	    assert_eq!(x+y,z)
	}
    }

    {
	const M:usize = 1920;
	const N:usize = 1080;
	let mut rng = Pcg64::from_entropy();
	let uniform_dist:Uniform<f32> = Uniform::new(-2.0,2.0);
	let context = cuda_tensor::Context::new(cuda_context)?;

	let tensor_x_src = (0..M*N).map(|_| uniform_dist.sample(&mut rng) ).collect::<Vec<f32>>();
	let tensor_y_src = (0..M*N).map(|_| uniform_dist.sample(&mut rng) ).collect::<Vec<f32>>();

	let mut tensor_x = context.create_tensor::<f32>(&[M,N])?;
	tensor_x.set_elements(tensor_x_src.clone())?;
	let mut tensor_y = context.create_tensor::<f32>(&[M,N])?;
	tensor_y.set_elements(tensor_y_src.clone())?;
	let tensor_z = cuda_tensor::tensor::add(&tensor_x, &tensor_y);
	let tensor_z_host = tensor_z.get_elements()?;

	for ((&x,&y),&z) in tensor_x_src.iter().zip(tensor_y_src.iter()).zip(tensor_z_host.iter())
	{
	    assert_eq!(x+y,z)
	}
    }

    Ok(())
}
